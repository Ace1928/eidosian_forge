from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def SHIFT_FREQUENCY(frame: Frame, freq: ParameterDesignator) -> ShiftFrequency:
    """
    Produce a SHIFT-FREQUENCY instruction.

    :param frame: The frame on which to shift the frequency.
    :param freq: The value, in Hz, to add to the existing frequency.
    :returns: A ShiftFrequency instance.
    """
    return ShiftFrequency(frame, freq)