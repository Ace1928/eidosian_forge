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
def _validate_weights(w, dtype=np.float64):
    w = _validate_vector(w, dtype=dtype)
    if np.any(w < 0):
        raise ValueError('Input weights should be all non-negative')
    return w