import asyncio
import collections
import logging
import time
import aiokafka.errors as Errors
from aiokafka.client import ConnectionGroup, CoordinationType
from aiokafka.errors import (
from aiokafka.protocol.produce import ProduceRequest
from aiokafka.protocol.transaction import (
from aiokafka.structs import TopicPartition
from aiokafka.util import create_task
class SendProduceReqHandler(BaseHandler):

    def __init__(self, sender, batches):
        super().__init__(sender)
        self._batches = batches
        self._client = sender.client
        self._to_reenqueue = []

    def create_request(self):
        topics = collections.defaultdict(list)
        for tp, batch in self._batches.items():
            topics[tp.topic].append((tp.partition, batch.get_data_buffer()))
        if self._client.api_version >= (2, 1):
            version = 7
        elif self._client.api_version >= (2, 0):
            version = 6
        elif self._client.api_version >= (1, 1):
            version = 5
        elif self._client.api_version >= (1, 0):
            version = 4
        elif self._client.api_version >= (0, 11):
            version = 3
        elif self._client.api_version >= (0, 10):
            version = 2
        elif self._client.api_version == (0, 9):
            version = 1
        else:
            version = 0
        kwargs = {}
        if version >= 3:
            if self._sender._txn_manager is not None:
                kwargs['transactional_id'] = self._sender._txn_manager.transactional_id
            else:
                kwargs['transactional_id'] = None
        request = ProduceRequest[version](required_acks=self._sender._acks, timeout=self._sender._request_timeout_ms, topics=list(topics.items()), **kwargs)
        return request

    async def do(self, node_id):
        request = self.create_request()
        try:
            response = await self._client.send(node_id, request)
        except KafkaError as err:
            log.warning('Got error produce response: %s', err)
            if getattr(err, 'invalid_metadata', False):
                self._client.force_metadata_update()
            for batch in self._batches.values():
                if not self._can_retry(err, batch):
                    batch.failure(exception=err)
                else:
                    self._to_reenqueue.append(batch)
        else:
            if request.required_acks == 0:
                for batch in self._batches.values():
                    batch.done_noack()
            else:
                self.handle_response(response)
        if self._to_reenqueue:
            await asyncio.sleep(self._default_backoff)
            for batch in self._to_reenqueue:
                self._sender._message_accumulator.reenqueue(batch)
            await self._client._maybe_wait_metadata()

    def handle_response(self, response):
        for topic, partitions in response.topics:
            for partition_info in partitions:
                global_error = None
                log_start_offset = None
                if response.API_VERSION < 2:
                    partition, error_code, offset = partition_info
                    timestamp = -1
                elif 2 <= response.API_VERSION <= 4:
                    partition, error_code, offset, timestamp = partition_info
                elif 5 <= response.API_VERSION <= 7:
                    partition, error_code, offset, timestamp, log_start_offset = partition_info
                else:
                    partition, error_code, offset, timestamp, log_start_offset, _, global_error = partition_info
                tp = TopicPartition(topic, partition)
                error = Errors.for_code(error_code)
                batch = self._batches.get(tp)
                if batch is None:
                    continue
                if error is Errors.NoError:
                    batch.done(offset, timestamp, log_start_offset)
                elif error is DuplicateSequenceNumber:
                    batch.done(offset, timestamp, log_start_offset)
                elif not self._can_retry(error(), batch):
                    if error is InvalidProducerEpoch:
                        exc = ProducerFenced()
                    elif error is TopicAuthorizationFailedError:
                        exc = error(topic)
                    else:
                        exc = error()
                    batch.failure(exception=exc)
                else:
                    log.warning('Got error produce response on topic-partition %s, retrying. Error: %s', tp, global_error or error)
                    if getattr(error, 'invalid_metadata', False):
                        self._client.force_metadata_update()
                    self._to_reenqueue.append(batch)

    def _can_retry(self, error, batch):
        if self._sender._txn_manager is None and batch.expired():
            return False
        if error.retriable or isinstance(error, UnknownTopicOrPartitionError) or error is UnknownTopicOrPartitionError:
            return True
        return False