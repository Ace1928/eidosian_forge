from dataclasses import dataclass
from typing import Generic, TypeVar
@dataclass
class DataConsumer(Generic[ConsumerType]):
    """A data class for representating a consumer of an output of a module."""
    consumer: ConsumerType
    consumer_input_idx: int
    output_idx: int