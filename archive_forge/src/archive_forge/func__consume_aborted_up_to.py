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
def _consume_aborted_up_to(self, batch_offset):
    aborted_transactions = self._aborted_transactions
    while aborted_transactions:
        producer_id, first_offset = aborted_transactions[0]
        if first_offset <= batch_offset:
            self._aborted_producers.add(producer_id)
            aborted_transactions.pop(0)
        else:
            break