import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_udf_array_binary(binary_func_fixture):
    check_scalar_function(binary_func_fixture, [pa.array([10, 20], pa.int64()), pa.array([2, 4], pa.int64())])