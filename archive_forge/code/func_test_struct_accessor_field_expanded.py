import re
import pytest
from pandas.compat.pyarrow import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize(['indices', 'name'], [(0, 'int_col'), ([1, 2], 'str_col'), (pc.field('int_col'), 'int_col'), ('int_col', 'int_col'), (b'string_col', b'string_col'), ([b'string_col'], 'string_col')])
@pytest.mark.skipif(pa_version_under13p0, reason='pyarrow>=13.0.0 required')
def test_struct_accessor_field_expanded(indices, name):
    arrow_type = pa.struct([('int_col', pa.int64()), ('struct_col', pa.struct([('int_col', pa.int64()), ('float_col', pa.float64()), ('str_col', pa.string())])), (b'string_col', pa.string())])
    data = pa.array([], type=arrow_type)
    ser = Series(data, dtype=ArrowDtype(arrow_type))
    expected = pc.struct_field(data, indices)
    result = ser.struct.field(indices)
    tm.assert_equal(result.array._pa_array.combine_chunks(), expected)
    assert result.name == name