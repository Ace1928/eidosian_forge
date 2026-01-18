from __future__ import annotations
import pickle
from datetime import timedelta
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import (
from xarray.tests import (
from xarray.tests.test_coding_times import (
@requires_cftime
@pytest.mark.parametrize('method', ['floor', 'ceil', 'round'])
def test_rounding_methods_invalid_freq(method):
    index = xr.cftime_range('2000-01-02T01:03:51', periods=10, freq='1777s')
    with pytest.raises(ValueError, match='fixed'):
        getattr(index, method)('MS')