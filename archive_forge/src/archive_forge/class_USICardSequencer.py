import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class USICardSequencer(Message):
    """
    Configuration for the card which
      interfaces with the USI Target on the USRP.
    """
    tx_channel: str
    'The label of the associated channel.'