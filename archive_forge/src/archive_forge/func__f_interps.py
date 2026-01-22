import warnings
import numpy as np
import scipy.interpolate
import scipy.signal
from ..util.exceptions import ParameterError
from ..util import is_unique
from numpy.typing import ArrayLike
from typing import Callable, Optional, Sequence
def _f_interps(data, f):
    interp = scipy.interpolate.interp1d(freqs[idx], data[idx], axis=0, bounds_error=False, copy=False, assume_sorted=False, kind=kind, fill_value=fill_value)
    return interp(f)