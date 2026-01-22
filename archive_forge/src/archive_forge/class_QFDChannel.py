import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class QFDChannel(Message):
    """
    Configuration for a single QFD Channel.
    """
    channel_index: int
    'The channel index on the QFD, zero indexed from the\n          lowest channel, as installed in the box.'
    direction: Optional[str] = 'tx'
    'The QFD is a device that transmits pulses.'
    nco_frequency: Optional[float] = 0.0
    'The DAC NCO frequency [Hz].'
    gain: Optional[float] = 0.0
    'The output gain on the DAC in [dB]. Note that this\n          should be in the range -45dB to 0dB and is rounded to the\n          nearest 3dB step.'
    delay: float = 0.0
    'Delay [seconds] to account for inter-channel skew.'
    flux_current: Optional[float] = None
    'Flux current [Amps].'
    relay_closed: Optional[bool] = None
    'Set the state of the Flux relay.\n          True  - Relay closed, allows flux current to flow.\n          False - Relay open, no flux current can flow.'