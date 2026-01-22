import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class QRTChannel(Message):
    """
    Configuration for a single QRT Channel.
    """
    direction: Optional[str] = 'tx'
    'The QRT is a device that transmits readout pulses.'
    nco_frequency: Optional[float] = 1250000000.0
    'The DAC NCO frequency [Hz].'
    gain: Optional[float] = 0.0
    'The output gain on the DAC in [dB]. Note that this should be in the range\n       -45dB to 0dB and is rounded to the nearest 3dB step.'
    channel_index: Optional[int] = 0
    'The channel index on the QRT, zero indexed from the lowest channel,\n        as installed in the box.'
    delay: float = 0.0
    'Delay [seconds] to account for inter-channel skew.'