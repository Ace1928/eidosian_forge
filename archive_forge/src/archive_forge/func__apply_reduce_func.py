from collections.abc import Sequence
import numpy as np
import xarray as xr
from numpy.linalg import LinAlgError
from scipy import special, stats
from . import _remove_indexes_to_reduce
from .linalg import cholesky, eigh
def _apply_reduce_func(func, da, dims, kwargs, func_kwargs=None):
    """Help wrap functions with a single input that return an output after reducing some dimensions.

    It assumes that the function to be applied only takes ``int`` as ``axis`` and stacks multiple
    dimensions if necessary to support reducing multiple dimensions at once.
    """
    if dims is None:
        dims = get_default_dims(da.dims)
    if not isinstance(dims, str):
        aux_dim = f'__aux_dim__:{','.join(dims)}'
        da = _remove_indexes_to_reduce(da, dims).stack({aux_dim: dims}, create_index=False)
        core_dims = [aux_dim]
    else:
        core_dims = [dims]
    out_da = xr.apply_ufunc(func, da, input_core_dims=[core_dims], output_core_dims=[[]], kwargs=func_kwargs, **kwargs)
    return out_da