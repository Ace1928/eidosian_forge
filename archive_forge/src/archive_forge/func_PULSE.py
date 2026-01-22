from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def PULSE(frame: Frame, waveform: Waveform, nonblocking: bool=False) -> Pulse:
    """
    Produce a PULSE instruction.

    :param frame: The frame on which to apply the pulse.
    :param waveform: The pulse waveform.
    :param nonblocking: A flag indicating whether the pulse is NONBLOCKING.
    :return: A Pulse instance.
    """
    return Pulse(frame, waveform, nonblocking)