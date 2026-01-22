from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def SET_FREQUENCY(frame: Frame, freq: ParameterDesignator) -> SetFrequency:
    """
    Produce a SET-FREQUENCY instruction.

    :param frame: The frame on which to set the frequency.
    :param freq: The frequency value, in Hz.
    :returns: A SetFrequency instance.
    """
    return SetFrequency(frame, freq)