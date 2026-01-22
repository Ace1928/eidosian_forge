import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class MNIOConnection(Message):
    """
    Description of one side of an MNIO connection between two Tsunamis.
    """
    port: int
    'The physical Tsunami MNIO port, indexed from 0, \n          where this connection originates.'
    destination: str
    'The Tsunami where this connection terminates.'