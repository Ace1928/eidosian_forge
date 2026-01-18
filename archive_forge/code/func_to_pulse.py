from __future__ import annotations
import functools
import textwrap
import pydoc
from collections.abc import Callable
import numpy as np
from ...exceptions import PulseError
from ..waveform import Waveform
from . import strategies
@functools.wraps(func)
def to_pulse(duration, *args, name=None, **kwargs):
    """Return Waveform."""
    if isinstance(duration, (int, np.integer)) and duration > 0:
        samples = func(duration, *args, **kwargs)
        samples = np.asarray(samples, dtype=np.complex128)
        return Waveform(samples=samples, name=name)
    raise PulseError('The first argument must be an integer value representing duration.')