import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class QDOFastFluxChannel(Message):
    """
    Configuration for a single QDO Fast Flux Channel.
    """
    channel_index: int
    'The channel index on the QDO, zero indexed from the\n          lowest channel, as installed in the box.'
    direction: Optional[str] = 'tx'
    'The QDO is a device that transmits pulses.'
    delay: float = 0.0
    'Delay [seconds] to account for inter-channel skew.'
    flux_current: Optional[float] = None
    'Flux current [Amps].'