from itertools import product, tee
import numpy as np
import xarray as xr
from .labels import BaseLabeller
def xarray_var_iter(data, var_names=None, combined=False, skip_dims=None, reverse_selections=False, dim_order=None):
    """Convert xarray data to an iterator over vectors.

    Iterates over each var_name and all of its coordinates, returning the 1d
    data.

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

    dim_order: list
        Order for the first dimensions. Skips dimensions not found in the variable.

    Returns
    -------
    Iterator of (str, dict(str, any), np.array)
        The string is the variable name, the dictionary are coordinate names to values,
        and the array are the values of the variable at those coordinates.
    """
    data_to_sel = data
    if var_names is None and isinstance(data, xr.DataArray):
        data_to_sel = {data.name: data}
    if isinstance(dim_order, str):
        dim_order = [dim_order]
    for var_name, selection, iselection in xarray_sel_iter(data, var_names=var_names, combined=combined, skip_dims=skip_dims, reverse_selections=reverse_selections):
        selected_data = data_to_sel[var_name].sel(**selection)
        if dim_order is not None:
            dim_order_selected = [dim for dim in dim_order if dim in selected_data.dims]
            if dim_order_selected:
                selected_data = selected_data.transpose(*dim_order_selected, ...)
        yield (var_name, selection, iselection, selected_data.values)