from itertools import product, tee
import numpy as np
import xarray as xr
from .labels import BaseLabeller
def xarray_to_ndarray(data, *, var_names=None, combined=True, label_fun=None):
    """Take xarray data and unpacks into variables and data into list and numpy array respectively.

    Assumes that chain and draw are in coordinates

    Parameters
    ----------
    data: xarray.DataSet
        Data in an xarray from an InferenceData object. Examples include posterior or sample_stats

    var_names: iter
        Should be a subset of data.data_vars not including chain and draws. Defaults to all of them

    combined: bool
        Whether to combine chain into one array

    Returns
    -------
    var_names: list
        List of variable names
    data: np.array
        Data values
    """
    if label_fun is None:
        label_fun = BaseLabeller().make_label_vert
    data_to_sel = data
    if var_names is None and isinstance(data, xr.DataArray):
        data_to_sel = {data.name: data}
    iterator1, iterator2 = tee(xarray_sel_iter(data, var_names=var_names, combined=combined))
    vars_and_sel = list(iterator1)
    unpacked_var_names = [label_fun(var_name, selection, isel) for var_name, selection, isel in vars_and_sel]
    data0 = data_to_sel[vars_and_sel[0][0]].sel(**vars_and_sel[0][1])
    unpacked_data = np.empty((len(unpacked_var_names), data0.size), dtype=data0.dtype)
    for idx, (var_name, selection, _) in enumerate(iterator2):
        unpacked_data[idx] = data_to_sel[var_name].sel(**selection).values.flatten()
    return (unpacked_var_names, unpacked_data)