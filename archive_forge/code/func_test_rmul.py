from __future__ import annotations
from itertools import product
from typing import Callable, Literal
import numpy as np
import pandas as pd
import pytest
from xarray import CFTimeIndex
from xarray.coding.cftime_offsets import (
from xarray.coding.frequencies import infer_freq
from xarray.core.dataarray import DataArray
from xarray.tests import (
@pytest.mark.parametrize(('offset', 'multiple', 'expected'), _MUL_TESTS, ids=_id_func)
def test_rmul(offset, multiple, expected):
    assert multiple * offset == expected