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
def test_xarray_sel_iter_ordering():
    """Assert that coordinate names stay the provided order"""
    coords = list('dcba')
    data = from_dict({'x': np.random.randn(1, 100, len(coords))}, coords={'in_order': coords}, dims={'x': ['in_order']}).posterior
    coord_names = [sel['in_order'] for _, sel, _ in xarray_sel_iter(data)]
    assert coord_names == coords