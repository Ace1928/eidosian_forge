import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_udf_array_varargs(varargs_func_fixture):
    check_scalar_function(varargs_func_fixture, [pa.array([2, 3], pa.int64()), pa.array([10, 20], pa.int64()), pa.array([3, 7], pa.int64()), pa.array([20, 30], pa.int64()), pa.array([5, 10], pa.int64())])