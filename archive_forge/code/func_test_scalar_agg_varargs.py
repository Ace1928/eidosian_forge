import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_scalar_agg_varargs(varargs_agg_func_fixture):
    arr1 = pa.array([10, 20, 30, 40, 50], pa.int64())
    arr2 = pa.array([1.0, 2.0, 3.0, 4.0, 5.0], pa.float64())
    result = pc.call_function('sum_mean', [arr1, arr2])
    expected = pa.scalar(33.0)
    assert result == expected