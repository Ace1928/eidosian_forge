import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class BinaryExecutableRequest(Message):
    """
    Native Quil and the information needed to create binary executables.
    """
    quil: str
    'Native Quil to be translated into an executable program.'
    num_shots: int
    'The number of times to repeat the program.'