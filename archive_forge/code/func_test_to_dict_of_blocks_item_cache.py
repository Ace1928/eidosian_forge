import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
def test_to_dict_of_blocks_item_cache(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': [1, 2, 3, 4], 'b': ['a', 'b', 'c', 'd']})
    df['c'] = NumpyExtensionArray(np.array([1, 2, None, 3], dtype=object))
    mgr = df._mgr
    assert len(mgr.blocks) == 3
    ser = df['b']
    df._to_dict_of_blocks()
    if using_copy_on_write:
        with pytest.raises(ValueError, match='read-only'):
            ser.values[0] = 'foo'
    elif warn_copy_on_write:
        ser.values[0] = 'foo'
        assert df.loc[0, 'b'] == 'foo'
        assert df['b'] is not ser
    else:
        ser.values[0] = 'foo'
        assert df.loc[0, 'b'] == 'foo'
        assert df['b'] is ser