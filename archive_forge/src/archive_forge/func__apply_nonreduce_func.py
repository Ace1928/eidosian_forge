from collections.abc import Sequence
import numpy as np
import xarray as xr
from numpy.linalg import LinAlgError
from scipy import special, stats
from . import _remove_indexes_to_reduce
from .linalg import cholesky, eigh
def _apply_nonreduce_func(func, da, dims, kwargs, func_kwargs=None):
    """Help wrap functions with a single input that return an output with the same size."""
    unstack = False
    if dims is None:
        dims = get_default_dims(da.dims)
    if not isinstance(dims, str):
        aux_dim = f'__aux_dim__:{','.join(dims)}'
        da = _remove_indexes_to_reduce(da, dims).stack({aux_dim: dims})
        core_dims = [aux_dim]
        unstack = True
    else:
        core_dims = [dims]
    out_da = xr.apply_ufunc(func, da, input_core_dims=[core_dims], output_core_dims=[core_dims], kwargs=func_kwargs, **kwargs)
    if unstack:
        return _remove_indexes_to_reduce(out_da.unstack(aux_dim), dims).reindex_like(da)
    return out_da