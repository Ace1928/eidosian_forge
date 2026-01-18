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
@pytest.mark.parametrize('c_values', [['blue', 'blue'], ['blue', '#0000ff'], np.array([[0, 0, 1], [0, 0, 1]])])
def test_vectorized_to_hex_array(c_values):
    output = vectorized_to_hex(c_values)
    assert np.all([item == '#0000ff' for item in output])