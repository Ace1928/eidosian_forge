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
def __last_beat(cumscore):
    """Get the last beat from the cumulative score array"""
    maxes = util.localmax(cumscore)
    med_score = np.median(cumscore[np.argwhere(maxes)])
    return np.argwhere(cumscore * maxes * 2 > med_score).max()