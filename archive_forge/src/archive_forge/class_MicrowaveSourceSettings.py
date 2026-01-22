import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class MicrowaveSourceSettings(Message):
    """
    Configuration of Microwave Source settings for operating amplifiers.
    """
    frequency: float
    'Frequency setting for microwave source (Hz).'
    power: float
    'Power setting for microwave source (dBm).'
    output: bool
    'Output setting for microwave source. If true, the source will be turned on.'