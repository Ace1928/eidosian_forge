import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class CWFrequencySweep(Message):
    """
    Configuration of a continuous wave frequency sweep.
    """
    start: float
    'Start frequency of the sweep, in Hz'
    stop: float
    'Stop frequency of the sweep, in Hz'
    num_pts: int
    'Number of frequency points to sample, cast to int.'
    source: int
    'Source port number'
    measure: int
    'Measure port number'