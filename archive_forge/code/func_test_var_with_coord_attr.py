from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from xarray import (
from xarray.backends.common import WritableCFDataStore
from xarray.backends.memory import InMemoryDataStore
from xarray.conventions import decode_cf
from xarray.testing import assert_identical
from xarray.tests import (
from xarray.tests.test_backends import CFEncodedBase
def test_var_with_coord_attr(self) -> None:
    orig = Dataset({'values': ('time', np.zeros(2), {'coordinates': 'time lon lat'})}, coords={'time': ('time', np.zeros(2)), 'lat': ('time', np.zeros(2)), 'lon': ('time', np.zeros(2))})
    enc, attrs = conventions.encode_dataset_coordinates(orig)
    values_coords = enc['values'].attrs.get('coordinates', '')
    assert values_coords == 'time lon lat'
    assert 'coordinates' not in attrs