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
class OffsetResetStrategy:
    LATEST = -1
    EARLIEST = -2
    NONE = 0

    @classmethod
    def from_str(cls, name):
        name = name.lower()
        if name == 'latest':
            return cls.LATEST
        if name == 'earliest':
            return cls.EARLIEST
        if name == 'none':
            return cls.NONE
        else:
            log.warning('Unrecognized ``auto_offset_reset`` config, using NONE')
            return cls.NONE

    @classmethod
    def to_str(cls, value):
        if value == cls.LATEST:
            return 'latest'
        if value == cls.EARLIEST:
            return 'earliest'
        if value == cls.NONE:
            return 'none'
        else:
            return f'timestamp({value})'