import typing
from typing import NamedTuple, Optional
class AckRequest(NamedTuple):
    ack_id: str
    byte_size: int
    time_to_ack: float
    ordering_key: Optional[str]
    future: Optional['futures.Future']