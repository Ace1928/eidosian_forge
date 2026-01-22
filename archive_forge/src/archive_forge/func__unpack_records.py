import asyncio
import collections
import logging
import random
import time
from itertools import chain
import async_timeout
import aiokafka.errors as Errors
from aiokafka.errors import (
from aiokafka.protocol.offset import OffsetRequest
from aiokafka.protocol.fetch import FetchRequest
from aiokafka.record.memory_records import MemoryRecords
from aiokafka.record.control_record import ControlRecord, ABORT_MARKER
from aiokafka.structs import OffsetAndTimestamp, TopicPartition, ConsumerRecord
from aiokafka.util import create_future, create_task
def _unpack_records(self):
    tp = self._tp
    records = self._records
    while records.has_next():
        next_batch = records.next_batch()
        if self._check_crcs and (not next_batch.validate_crc()):
            raise Errors.CorruptRecordException(f'Invalid CRC - {tp}')
        if self._isolation_level == READ_COMMITTED and next_batch.producer_id is not None:
            self._consume_aborted_up_to(next_batch.base_offset)
            if next_batch.is_control_batch:
                if self._contains_abort_marker(next_batch):
                    self._aborted_producers.discard(next_batch.producer_id)
            if next_batch.is_transactional and next_batch.producer_id in self._aborted_producers:
                log.debug('Skipping aborted record batch from partition %s with producer_id %s and offsets %s to %s', tp, next_batch.producer_id, next_batch.base_offset, next_batch.next_offset - 1)
                self.next_fetch_offset = next_batch.next_offset
                continue
        if next_batch.is_control_batch:
            self.next_fetch_offset = next_batch.next_offset
            continue
        for record in next_batch:
            if record.offset < self.next_fetch_offset:
                continue
            consumer_record = self._consumer_record(tp, record)
            self.next_fetch_offset = record.offset + 1
            yield consumer_record
        self.next_fetch_offset = next_batch.next_offset