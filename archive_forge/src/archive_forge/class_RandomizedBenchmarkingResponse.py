import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class RandomizedBenchmarkingResponse(Message):
    """
    RPC reply payload for a randomly generated benchmarking sequence.
    """
    sequence: List[List[int]]
    'List of Cliffords, each expressed as a list of generator indices.'