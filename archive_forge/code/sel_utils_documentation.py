from itertools import product, tee
import numpy as np
import xarray as xr
from .labels import BaseLabeller
Take xarray data and unpacks into variables and data into list and numpy array respectively.

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
    