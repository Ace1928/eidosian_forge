from dataclasses import dataclass
from typing import Generic, List, NamedTuple, Optional, Sequence, Tuple, TypeVar
from aiokafka.errors import KafkaError
class RecordMetadata(NamedTuple):
    """Returned when a :class:`~.AIOKafkaProducer` sends a message"""
    topic: str
    'The topic name'
    partition: int
    'The partition number'
    topic_partition: TopicPartition
    ''
    offset: int
    'The unique offset of the message in this partition.\n\n    See :ref:`Offsets and Consumer Position <offset_and_position>` for more\n    details on offsets.\n    '
    timestamp: Optional[int]
    'Timestamp in millis, None for older Brokers'
    timestamp_type: int
    "The timestamp type of this record.\n\n    If the broker respected the timestamp passed to\n    :meth:`.AIOKafkaProducer.send`, ``0`` will be returned (``CreateTime``).\n\n    If the broker set it's own timestamp, ``1`` will be returned (``LogAppendTime``).\n    "
    log_start_offset: Optional[int]
    ''