import warnings
import cupy
from cupy import _core
from cupy._core import _routines_statistics as _statistics
from cupy._core import _fusion_thread_local
from cupy._logic import content
def _quantile_is_valid(q):
    if cupy.count_nonzero(q < 0.0) or cupy.count_nonzero(q > 1.0):
        return False
    return True