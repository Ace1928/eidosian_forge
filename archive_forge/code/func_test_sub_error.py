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
@pytest.mark.parametrize('offset', _EQ_TESTS_A, ids=_id_func)
def test_sub_error(offset, calendar):
    date_type = get_date_type(calendar)
    initial = date_type(1, 1, 1)
    with pytest.raises(TypeError):
        offset - initial