import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
def test_unhashable_map_keys_with_pydicts():
    keys = pa.array([['a', 'b'], ['c', 'd'], [], ['e'], [None, 'f'], ['g', 'h']], pa.list_(pa.string()))
    items = pa.array(['foo', 'bar', 'baz', 'qux', 'quux', 'quz'])
    offsets = [0, 2, 6]
    maps = pa.MapArray.from_arrays(offsets, keys, items)
    with pytest.raises(TypeError):
        maps.to_pandas(maps_as_pydicts='lossy')
    series = maps.to_pandas()
    expected_series_default = pd.Series([[(['a', 'b'], 'foo'), (['c', 'd'], 'bar')], [([], 'baz'), (['e'], 'qux'), ([None, 'f'], 'quux'), (['g', 'h'], 'quz')]])
    assert len(series) == len(expected_series_default)
    for row1, row2 in zip(series, expected_series_default):
        assert len(row1) == len(row2)
        for tup1, tup2 in zip(row1, row2):
            assert np.array_equal(tup1[0], tup2[0])
            assert tup1[1] == tup2[1]