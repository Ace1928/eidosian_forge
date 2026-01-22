import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class QuiltCalibrationsResponse(Message):
    """
    Up-to-date Quilt calibrations.
    """
    quilt: str
    'Quilt code with definitions for frames, waveforms, and calibrations.'