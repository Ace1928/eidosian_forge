from dataclasses import dataclass
from typing import Generic, List, NamedTuple, Optional, Sequence, Tuple, TypeVar
from aiokafka.errors import KafkaError
class PartitionMetadata(NamedTuple):
    """A topic partition metadata describing the state in the MetadataResponse"""
    topic: str
    'The topic name of the partition this metadata relates to'
    partition: int
    'The id of the partition this metadata relates to'
    leader: int
    'The id of the broker that is the leader for the partition'
    replicas: List[int]
    'The ids of all brokers that contain replicas of the partition'
    isr: List[int]
    'The ids of all brokers that contain in-sync replicas of the partition'
    error: Optional[KafkaError]
    'A KafkaError object associated with the request for this partition metadata'