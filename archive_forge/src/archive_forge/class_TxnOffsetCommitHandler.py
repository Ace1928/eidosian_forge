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
class TxnOffsetCommitHandler(BaseHandler):
    group = ConnectionGroup.COORDINATION

    def __init__(self, sender, offsets, group_id):
        super().__init__(sender)
        self._offsets = offsets
        self._group_id = group_id

    def create_request(self):
        txn_manager = self._sender._txn_manager
        offset_data = collections.defaultdict(list)
        for tp, offset in sorted(self._offsets.items()):
            offset_data[tp.topic].append((tp.partition, offset.offset, offset.metadata))
        req = TxnOffsetCommitRequest[0](transactional_id=txn_manager.transactional_id, group_id=self._group_id, producer_id=txn_manager.producer_id, producer_epoch=txn_manager.producer_epoch, topics=list(offset_data.items()))
        return req

    def handle_response(self, resp):
        txn_manager = self._sender._txn_manager
        group_id = self._group_id
        for topic, partitions in resp.errors:
            for partition, error_code in partitions:
                tp = TopicPartition(topic, partition)
                error_type = Errors.for_code(error_code)
                if error_type is Errors.NoError:
                    offset = self._offsets[tp].offset
                    log.debug('Offset %s for partition %s committed to group %s', offset, tp, group_id)
                    txn_manager.offset_committed(tp, offset, group_id)
                elif error_type is CoordinatorNotAvailableError or error_type is NotCoordinatorError or error_type is RequestTimedOutError:
                    self._sender._coordinator_dead(CoordinationType.GROUP)
                    return self._default_backoff
                elif error_type is CoordinatorLoadInProgressError or error_type is UnknownTopicOrPartitionError:
                    return self._default_backoff
                elif error_type is InvalidProducerEpoch:
                    raise ProducerFenced()
                elif error_type is TransactionalIdAuthorizationFailed:
                    raise error_type(txn_manager.transactional_id)
                elif error_type is GroupAuthorizationFailedError:
                    exc = error_type(self._group_id)
                    txn_manager.error_transaction(exc)
                    return
                else:
                    log.error('Could not commit offset for partition %s due to unexpected error: %s', partition, error_type)
                    raise error_type()