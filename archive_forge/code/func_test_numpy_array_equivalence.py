from __future__ import annotations
from typing import Any
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_numpy_array_equivalence(data, all_arithmetic_operators):
    data, scalar = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)
    numpy_array = np.array([scalar] * len(data), dtype=data.dtype.numpy_dtype)
    pd_array = pd.array(numpy_array, dtype=data.dtype)
    if is_bool_not_implemented(data, all_arithmetic_operators):
        msg = "operator '.*' not implemented for bool dtypes"
        with pytest.raises(NotImplementedError, match=msg):
            op(data, numpy_array)
        with pytest.raises(NotImplementedError, match=msg):
            op(data, pd_array)
        return
    result = op(data, numpy_array)
    expected = op(data, pd_array)
    tm.assert_extension_array_equal(result, expected)