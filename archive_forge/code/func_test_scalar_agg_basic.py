import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_scalar_agg_basic(unary_agg_func_fixture):
    arr = pa.array([10.0, 20.0, 30.0, 40.0, 50.0], pa.float64())
    result = pc.call_function('mean_udf', [arr])
    expected = pa.scalar(30.0)
    assert result == expected