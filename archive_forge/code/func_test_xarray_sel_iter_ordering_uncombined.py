import importlib
import numpy as np
import pytest
import xarray as xr
from ...data import from_dict
from ...plots.backends.matplotlib import dealiase_sel_kwargs, matplotlib_kwarg_dealiaser
from ...plots.plot_utils import (
from ...rcparams import rc_context
from ...sel_utils import xarray_sel_iter, xarray_to_ndarray
from ...stats.density_utils import get_bins
from ...utils import get_coords
from ..helpers import running_on_ci
def test_xarray_sel_iter_ordering_uncombined(sample_dataset):
    """Assert that varname order stays consistent when chains are not combined"""
    _, _, data = sample_dataset
    var_names = [(var, selection) for var, selection, _ in xarray_sel_iter(data, var_names=None)]
    assert len(var_names) == 4
    for var_name in var_names:
        assert var_name in [('mu', {'chain': 0}), ('mu', {'chain': 1}), ('tau', {'chain': 0}), ('tau', {'chain': 1})]