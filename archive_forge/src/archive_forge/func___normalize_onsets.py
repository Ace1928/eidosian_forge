import numpy as np
import scipy
import scipy.stats
from ._cache import cache
from . import core
from . import onset
from . import util
from .feature import tempogram, fourier_tempogram
from .feature import tempo as _tempo
from .util.exceptions import ParameterError
from .util.decorators import moved
from typing import Any, Callable, Optional, Tuple
def __normalize_onsets(onsets):
    """Map onset strength function into the range [0, 1]"""
    norm = onsets.std(ddof=1)
    if norm > 0:
        onsets = onsets / norm
    return onsets