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
def test_reindex_warning(self) -> None:
    data = create_test_data()
    with pytest.raises(ValueError):
        ind = xr.DataArray([0.0, 1.0], dims=['new_dim'], name='ind')
        data.reindex(dim2=ind)
    ind = xr.DataArray([0.0, 1.0], dims=['dim2'], name='ind')
    with warnings.catch_warnings(record=True) as ws:
        data.reindex(dim2=ind)
        assert len(ws) == 0