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
@pytest.mark.parametrize('freq', ['A', 'AS', 'Q', 'M', 'H', 'T', 'S', 'L', 'U', 'Y', 'A-MAY'])
def test_to_offset_deprecation_warning(freq):
    with pytest.warns(FutureWarning, match='is deprecated'):
        to_offset(freq)