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
class TestDecodeCFVariableWithArrayUnits:

    def test_decode_cf_variable_with_array_units(self) -> None:
        v = Variable(['t'], [1, 2, 3], {'units': np.array(['foobar'], dtype=object)})
        v_decoded = conventions.decode_cf_variable('test2', v)
        assert_identical(v, v_decoded)