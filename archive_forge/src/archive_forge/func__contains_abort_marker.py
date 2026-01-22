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
def _contains_abort_marker(self, next_batch):
    try:
        control_record = next(next_batch)
    except StopIteration:
        raise Errors.KafkaError('Control batch did not contain any records')
    return ControlRecord.parse(control_record.key) == ABORT_MARKER