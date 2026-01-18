from itertools import product, tee
import numpy as np
import xarray as xr
from .labels import BaseLabeller
def xarray_sel_iter(data, var_names=None, combined=False, skip_dims=None, reverse_selections=False):
    """Convert xarray data to an iterator over variable names and selections.

    Iterates over each var_name and all of its coordinates, returning the variable
    names and selections that allow properly obtain the data from ``data`` as desired.

    Parameters
    ----------
    data : xarray.Dataset
        Posterior data in an xarray

    var_names : iterator of strings (optional)
        Should be a subset of data.data_vars. Defaults to all of them.

    combined : bool
        Whether to combine chains or leave them separate

    skip_dims : set
        dimensions to not iterate over

    reverse_selections : bool
        Whether to reverse selections before iterating.

    Returns
    -------
    Iterator of (var_name: str, selection: dict(str, any))
        The string is the variable name, the dictionary are coordinate names to values,.
        To get the values of the variable at these coordinates, do
        ``data[var_name].sel(**selection)``.
    """
    if skip_dims is None:
        skip_dims = set()
    if combined:
        skip_dims = skip_dims.union({'chain', 'draw'})
    else:
        skip_dims.add('draw')
    if var_names is None:
        if isinstance(data, xr.Dataset):
            var_names = list(data.data_vars)
        elif isinstance(data, xr.DataArray):
            var_names = [data.name]
            data = {data.name: data}
    for var_name in var_names:
        if var_name in data:
            new_dims = _dims(data, var_name, skip_dims)
            vals = [list(dict.fromkeys(data[var_name][dim].values)) for dim in new_dims]
            dims = _zip_dims(new_dims, vals)
            idims = _zip_dims(new_dims, [range(len(v)) for v in vals])
            if reverse_selections:
                dims = reversed(dims)
                idims = reversed(idims)
            for selection, iselection in zip(dims, idims):
                yield (var_name, selection, iselection)