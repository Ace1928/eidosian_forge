from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
@pytest.mark.parametrize('reduct, expected', [('dim1', ['dim2', 'dim3', 'time', 'dim1']), ('dim2', ['dim3', 'time', 'dim1', 'dim2']), ('dim3', ['dim2', 'time', 'dim1', 'dim3']), ('time', ['dim2', 'dim3', 'dim1'])])
@pytest.mark.parametrize('func', ['cumsum', 'cumprod'])
def test_reduce_cumsum_test_dims(self, reduct, expected, func) -> None:
    data = create_test_data()
    with pytest.raises(ValueError, match="Dimensions \\('bad_dim',\\) not found in data dimensions"):
        getattr(data, func)(dim='bad_dim')
    actual = getattr(data, func)(dim=reduct).dims
    assert list(actual) == expected