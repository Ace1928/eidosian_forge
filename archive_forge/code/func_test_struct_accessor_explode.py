import re
import pytest
from pandas.compat.pyarrow import (
from pandas import (
import pandas._testing as tm
@pytest.mark.skipif(pa_version_under11p0, reason='pyarrow>=11.0.0 required')
def test_struct_accessor_explode():
    index = Index([-100, 42, 123])
    ser = Series([{'painted': 1, 'snapping': {'sea': 'green'}}, {'painted': 2, 'snapping': {'sea': 'leatherback'}}, {'painted': 3, 'snapping': {'sea': 'hawksbill'}}], dtype=ArrowDtype(pa.struct([('painted', pa.int64()), ('snapping', pa.struct([('sea', pa.string())]))])), index=index)
    actual = ser.struct.explode()
    expected = DataFrame({'painted': Series([1, 2, 3], index=index, dtype=ArrowDtype(pa.int64())), 'snapping': Series([{'sea': 'green'}, {'sea': 'leatherback'}, {'sea': 'hawksbill'}], index=index, dtype=ArrowDtype(pa.struct([('sea', pa.string())])))})
    tm.assert_frame_equal(actual, expected)