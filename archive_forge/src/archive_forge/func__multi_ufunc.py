import warnings
from collections.abc import Sequence
from copy import copy as _copy
from copy import deepcopy as _deepcopy
import numpy as np
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.interpolate import CubicSpline
from scipy.stats.mstats import mquantiles
from xarray import apply_ufunc
from .. import _log
from ..utils import conditional_jit, conditional_vect, conditional_dask
from .density_utils import histogram as _histogram
def _multi_ufunc(*args, out=None, out_shape=None, **kwargs):
    """General ufunc for multi-output function."""
    arys = args[:n_input]
    element_shape = arys[-1].shape[:-n_dims]
    if out is None:
        if out_shape is None:
            out = tuple((np.empty(element_shape) for _ in range(n_output)))
        else:
            out = tuple((np.empty((*element_shape, *out_shape[i])) for i in range(n_output)))
    elif check_shape:
        raise_error = False
        correct_shape = tuple((element_shape for _ in range(n_output)))
        if isinstance(out, tuple):
            out_shape = tuple((item.shape for item in out))
            if out_shape != correct_shape:
                raise_error = True
        else:
            raise_error = True
            out_shape = 'not tuple, type={type(out)}'
        if raise_error:
            msg = f'Shapes incorrect for `out`: {out_shape}.'
            msg += f' Correct shapes are {correct_shape}'
            raise TypeError(msg)
    for idx in np.ndindex(element_shape):
        arys_idx = [ary[idx].ravel() if ravel else ary[idx] for ary in arys]
        results = func(*arys_idx, *args[n_input:], **kwargs)
        for i, res in enumerate(results):
            out[i][idx] = np.asarray(res)[index]
    return out