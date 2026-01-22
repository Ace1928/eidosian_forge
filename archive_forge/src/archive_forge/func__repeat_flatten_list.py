import warnings
from numbers import Integral
from itertools import repeat
import xarray as xr
import numpy as np
from xarray.core.dataarray import DataArray
from ..sel_utils import xarray_var_iter
from ..rcparams import rcParams
from .plot_utils import default_grid, filter_plotters_list, get_plotting_function
def _repeat_flatten_list(lst, n):
    return [item for sublist in repeat(lst, n) for item in sublist]