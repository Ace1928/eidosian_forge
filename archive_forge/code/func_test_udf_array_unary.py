import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_udf_array_unary(unary_func_fixture):
    check_scalar_function(unary_func_fixture, [pa.array([10, 20], pa.int64())])