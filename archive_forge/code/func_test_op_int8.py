import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('opname', ['floordiv', 'mod', 'pow'])
def test_op_int8(left_array, right_array, opname):
    op = getattr(operator, opname)
    if opname != 'mod':
        msg = "operator '.*' not implemented for bool dtypes"
        with pytest.raises(NotImplementedError, match=msg):
            result = op(left_array, right_array)
        return
    result = op(left_array, right_array)
    expected = op(left_array.astype('Int8'), right_array.astype('Int8'))
    tm.assert_extension_array_equal(result, expected)