import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
def _convert_fallback_to_numpy(args, kwargs):
    return _get_xp_args(ndarray, ndarray._get_numpy_array, (args, kwargs))