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
@requires_dask
def test_string_object_warning(self) -> None:
    original = Variable(('x',), np.array(['foo', 'bar'], dtype=object)).chunk()
    with pytest.warns(SerializationWarning, match='dask array with dtype=object'):
        encoded = conventions.encode_cf_variable(original)
    assert_identical(original, encoded)