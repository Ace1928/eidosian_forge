import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.mark.pandas
def test_vector_basic(unary_vector_func_fixture):
    arr = pa.array([10.0, 20.0, 30.0, 40.0, 50.0], pa.float64())
    result = pc.call_function('y=pct_rank(x)', [arr])
    expected = unary_vector_func_fixture[0](None, arr)
    assert result == expected