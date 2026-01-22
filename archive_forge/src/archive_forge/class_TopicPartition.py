from dataclasses import dataclass
from typing import Generic, List, NamedTuple, Optional, Sequence, Tuple, TypeVar
from aiokafka.errors import KafkaError
class TopicPartition(NamedTuple):
    """A topic and partition tuple"""
    topic: str
    'A topic name'
    partition: int
    'A partition id'