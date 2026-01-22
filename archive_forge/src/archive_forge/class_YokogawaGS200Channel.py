import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class YokogawaGS200Channel(Message):
    """
    Configuration for a single Yokogawa GS200 Channel.
    """