import re
import pytest
from pandas.compat.pyarrow import (
from pandas import (
import pandas._testing as tm
@pytest.mark.skipif(pa_version_under13p0, reason='pyarrow>=13.0.0 required')
def test_struct_accessor_field():
    index = Index([-100, 42, 123])
    ser = Series([{'rice': 1.0, 'maize': -1, 'wheat': 'a'}, {'rice': 2.0, 'maize': 0, 'wheat': 'b'}, {'rice': 3.0, 'maize': 1, 'wheat': 'c'}], dtype=ArrowDtype(pa.struct([('rice', pa.float64()), ('maize', pa.int64()), ('wheat', pa.string())])), index=index)
    by_name = ser.struct.field('maize')
    by_name_expected = Series([-1, 0, 1], dtype=ArrowDtype(pa.int64()), index=index, name='maize')
    tm.assert_series_equal(by_name, by_name_expected)
    by_index = ser.struct.field(2)
    by_index_expected = Series(['a', 'b', 'c'], dtype=ArrowDtype(pa.string()), index=index, name='wheat')
    tm.assert_series_equal(by_index, by_index_expected)