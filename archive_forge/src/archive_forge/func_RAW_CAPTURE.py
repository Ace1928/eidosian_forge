from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def RAW_CAPTURE(frame: Frame, duration: float, memory_region: MemoryReferenceDesignator, nonblocking: bool=False) -> RawCapture:
    """
    Produce a RAW-CAPTURE instruction.

    :param frame: The frame on which to capture raw values.
    :param duration: The duration of the capture, in seconds.
    :param memory_region: The classical memory region to store the resulting raw values.
    :param nonblocking: A flag indicating whether the capture is NONBLOCKING.
    :returns: A RawCapture instance.
    """
    memory_region = unpack_classical_reg(memory_region)
    return RawCapture(frame, duration, memory_region, nonblocking)