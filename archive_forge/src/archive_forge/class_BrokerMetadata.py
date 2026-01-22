from dataclasses import dataclass
from typing import Generic, List, NamedTuple, Optional, Sequence, Tuple, TypeVar
from aiokafka.errors import KafkaError
class BrokerMetadata(NamedTuple):
    """A Kafka broker metadata used by admin tools"""
    nodeId: int
    'The Kafka broker id'
    host: str
    'The Kafka broker hostname'
    port: int
    'The Kafka broker port'
    rack: Optional[str]
    'The rack of the broker, which is used to in rack aware partition\n    assignment for fault tolerance.\n    Examples: `RACK1`, `us-east-1d`. Default: None\n    '