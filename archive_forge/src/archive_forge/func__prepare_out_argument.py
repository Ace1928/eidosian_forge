import math
import warnings
import numpy as np
import dataclasses
from typing import Optional, Callable
from functools import partial
from scipy._lib._util import _asarray_validated
from . import _distance_wrap
from . import _hausdorff
from ..linalg import norm
from ..special import rel_entr
from . import _distance_pybind
def _prepare_out_argument(out, dtype, expected_shape):
    if out is None:
        return np.empty(expected_shape, dtype=dtype)
    if out.shape != expected_shape:
        raise ValueError('Output array has incorrect shape.')
    if not out.flags.c_contiguous:
        raise ValueError('Output array must be C-contiguous.')
    if out.dtype != np.float64:
        raise ValueError('Output array must be double type.')
    return out