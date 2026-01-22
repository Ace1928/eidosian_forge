import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class QuiltCalibrationsRequest(Message):
    """
    A request for up-to-date Quilt calibrations.
    """
    target_device: TargetDevice
    'Specifications for the device to get calibrations for.'