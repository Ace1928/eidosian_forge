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
def update_docstring(ufunc, func, n_output=1):
    """Update ArviZ generated ufunc docstring."""
    module = ''
    name = ''
    docstring = ''
    if hasattr(func, '__module__') and isinstance(func.__module__, str):
        module += func.__module__
    if hasattr(func, '__name__'):
        name += func.__name__
    if hasattr(func, '__doc__') and isinstance(func.__doc__, str):
        docstring += func.__doc__
    ufunc.__doc__ += '\n\n'
    if module or name:
        ufunc.__doc__ += 'This function is a ufunc wrapper for '
        ufunc.__doc__ += module + '.' + name
        ufunc.__doc__ += '\n'
    ufunc.__doc__ += 'Call ufunc with n_args from xarray against "chain" and "draw" dimensions:'
    ufunc.__doc__ += '\n\n'
    input_core_dims = 'tuple(("chain", "draw") for _ in range(n_args))'
    if n_output > 1:
        output_core_dims = f' tuple([] for _ in range({n_output}))'
        msg = f'xr.apply_ufunc(ufunc, dataset, input_core_dims={input_core_dims}, '
        msg += f'output_core_dims={output_core_dims})'
    else:
        output_core_dims = ''
        msg = f'xr.apply_ufunc(ufunc, dataset, input_core_dims={input_core_dims})'
    ufunc.__doc__ += msg
    ufunc.__doc__ += '\n\n'
    ufunc.__doc__ += 'For example: np.std(data, ddof=1) --> n_args=2'
    if docstring:
        ufunc.__doc__ += '\n\n'
        ufunc.__doc__ += module
        ufunc.__doc__ += name
        ufunc.__doc__ += ' docstring:'
        ufunc.__doc__ += '\n\n'
        ufunc.__doc__ += docstring