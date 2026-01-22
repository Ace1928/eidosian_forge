from __future__ import annotations
import numpy as np
from .channels import DriveChannel, MeasureChannel
from .exceptions import PulseError
def add_lo_range(self, channel: DriveChannel | MeasureChannel, lo_range: LoRange | tuple[int, int]):
    """Add lo range to configuration.

        Args:
            channel: Channel to add lo range for
            lo_range: Lo range to add

        """
    if isinstance(lo_range, (list, tuple)):
        lo_range = LoRange(*lo_range)
    self._lo_ranges[channel] = lo_range