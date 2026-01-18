from __future__ import annotations
from numbers import Number
import numpy as np
import pytest
import xarray as xr
from xarray.backends.api import _get_default_engine
from xarray.tests import (
@pytest.mark.parametrize('shape,pref_chunks,req_chunks', [((5,), (2,), (3,)), ((5,), (2,), ((2, 1, 1, 1),)), ((5,), ((2, 2, 1),), (3,)), ((5,), ((2, 2, 1),), ((2, 1, 1, 1),)), ((1, 5), (1, 2), (1, 3))])
def test_split_chunks(self, shape, pref_chunks, req_chunks):
    """Warn when the requested chunks separate the backend's preferred chunks."""
    initial = self.create_dataset(shape, pref_chunks)
    with pytest.warns(UserWarning):
        final = xr.open_dataset(initial, engine=PassThroughBackendEntrypoint, chunks=dict(zip(initial[self.var_name].dims, req_chunks)))
    self.check_dataset(initial, final, explicit_chunks(req_chunks, shape))