import asyncio
import collections
import copy
import time
from aiokafka.errors import (KafkaTimeoutError,
from aiokafka.record.legacy_records import LegacyRecordBatchBuilder
from aiokafka.record.default_records import DefaultRecordBatchBuilder
from aiokafka.structs import RecordMetadata
from aiokafka.util import create_future, get_running_loop
class MessageBatch:
    """This class incapsulate operations with batch of produce messages"""

    def __init__(self, tp, builder, ttl):
        self._builder = builder
        self._tp = tp
        self._ttl = ttl
        self._ctime = time.monotonic()
        self.future = create_future()
        self._msg_futures = []
        self._drain_waiter = create_future()
        self._retry_count = 0

    @property
    def tp(self):
        return self._tp

    @property
    def record_count(self):
        return self._builder.record_count()

    def append(self, key, value, timestamp_ms, _create_future=create_future, headers=[]):
        """Append message (key and value) to batch

        Returns:
            None if batch is full
              or
            asyncio.Future that will resolved when message is delivered
        """
        metadata = self._builder.append(timestamp=timestamp_ms, key=key, value=value, headers=headers)
        if metadata is None:
            return None
        future = _create_future()
        self._msg_futures.append((future, metadata))
        return future

    def done(self, base_offset, timestamp=None, log_start_offset=None, _record_metadata_class=RecordMetadata):
        """Resolve all pending futures"""
        tp = self._tp
        topic = tp.topic
        partition = tp.partition
        if timestamp == -1:
            timestamp_type = 0
        else:
            timestamp_type = 1
        if not self.future.done():
            self.future.set_result(_record_metadata_class(topic, partition, tp, base_offset, timestamp, timestamp_type, log_start_offset))
        for future, metadata in self._msg_futures:
            if future.done():
                continue
            if timestamp == -1:
                timestamp = metadata.timestamp
            offset = base_offset + metadata.offset
            future.set_result(_record_metadata_class(topic, partition, tp, offset, timestamp, timestamp_type, log_start_offset))

    def done_noack(self):
        """ Resolve all pending futures to None """
        if not self.future.done():
            self.future.set_result(None)
        for future, _ in self._msg_futures:
            if future.done():
                continue
            future.set_result(None)

    def failure(self, exception):
        if not self.future.done():
            self.future.set_exception(exception)
        for future, _ in self._msg_futures:
            if future.done():
                continue
            future.set_exception(copy.copy(exception))
        if self._msg_futures:
            self.future.exception()
        if not self._drain_waiter.done():
            self._drain_waiter.set_exception(exception)

    async def wait_drain(self, timeout=None):
        """Wait until all message from this batch is processed"""
        waiter = self._drain_waiter
        await asyncio.wait([waiter], timeout=timeout)
        if waiter.done():
            waiter.result()

    def expired(self):
        """Check that batch is expired or not"""
        return time.monotonic() - self._ctime > self._ttl

    def drain_ready(self):
        """Compress batch to be ready for send"""
        if not self._drain_waiter.done():
            self._drain_waiter.set_result(None)
        self._retry_count += 1

    def reset_drain(self):
        """Reset drain waiter, until we will do another retry"""
        assert self._drain_waiter.done()
        self._drain_waiter = create_future()

    def set_producer_state(self, producer_id, producer_epoch, base_sequence):
        assert not self._drain_waiter.done()
        self._builder._set_producer_state(producer_id, producer_epoch, base_sequence)

    def get_data_buffer(self):
        return self._builder._build()

    def is_empty(self):
        return self._builder.record_count() == 0

    @property
    def retry_count(self):
        return self._retry_count