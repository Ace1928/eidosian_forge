import warnings
import numpy as np
import scipy.interpolate
import scipy.signal
from ..util.exceptions import ParameterError
from ..util import is_unique
from numpy.typing import ArrayLike
from typing import Callable, Optional, Sequence
def _f_interp(_a, _b):
    interp = scipy.interpolate.interp1d(_a, _b, bounds_error=False, copy=False, kind=kind, fill_value=fill_value)
    return interp(np.multiply.outer(_a, harmonics))