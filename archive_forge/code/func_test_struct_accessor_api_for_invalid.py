import re
import pytest
from pandas.compat.pyarrow import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('invalid', [pytest.param(Series([1, 2, 3], dtype='int64'), id='int64'), pytest.param(Series(['a', 'b', 'c'], dtype='string[pyarrow]'), id='string-pyarrow')])
def test_struct_accessor_api_for_invalid(invalid):
    with pytest.raises(AttributeError, match=re.escape(f"Can only use the '.struct' accessor with 'struct[pyarrow]' dtype, not {invalid.dtype}.")):
        invalid.struct