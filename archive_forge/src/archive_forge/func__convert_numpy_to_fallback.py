import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
def _convert_numpy_to_fallback(numpy_res):
    return _get_xp_args(np.ndarray, ndarray._store_array_from_numpy, numpy_res)