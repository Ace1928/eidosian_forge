import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class CWChannel(Message):
    """
    Configuration for a single CW Generator Channel.
    """
    channel_index: int = 0
    "The zero-indexed channel of the generator's output."
    rf_output_frequency: Optional[int] = 1000000000
    "The CW generator's output frequency [Hz]."
    rf_output_power: Optional[float] = 0.0
    "The power of CW generator's output [dBm]."
    rf_output_enabled: Optional[bool] = True
    "The state (on/off) of CW generator's output."