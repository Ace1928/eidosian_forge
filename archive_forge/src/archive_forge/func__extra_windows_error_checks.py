import os
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
def _extra_windows_error_checks(x, out, required_shape, **kwargs):
    if os.name == 'nt' and out is not None:
        if out.shape != required_shape:
            raise ValueError('Output array has incorrect shape.')
        if not out.flags['C_CONTIGUOUS']:
            raise ValueError('Output array must be C-contiguous.')
        if not np.can_cast(x.dtype, out.dtype):
            raise ValueError('Wrong out dtype.')
    if os.name == 'nt' and 'w' in kwargs:
        w = kwargs['w']
        if w is not None:
            if (w < 0).sum() > 0:
                raise ValueError('Input weights should be all non-negative')