import asyncio
import collections
import copy
import time
from aiokafka.errors import (KafkaTimeoutError,
from aiokafka.record.legacy_records import LegacyRecordBatchBuilder
from aiokafka.record.default_records import DefaultRecordBatchBuilder
from aiokafka.structs import RecordMetadata
from aiokafka.util import create_future, get_running_loop
Add BatchBuilder to queue by topic-partition.

        Arguments:
            builder (BatchBuilder): batch object to enqueue.
            tp (TopicPartition): topic and partition to enqueue this batch for.
            timeout (int): time in seconds to wait for a free slot in the batch
                queue.

        Returns:
            MessageBatch: delivery wrapper around the BatchBuilder object.

        Raises:
            aiokafka.errors.ProducerClosed: the accumulator has already been
                closed and flushed.
            aiokafka.errors.KafkaTimeoutError: the batch could not be added
                within the specified timeout.
        