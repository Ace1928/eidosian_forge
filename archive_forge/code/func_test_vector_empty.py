import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.mark.pandas
def test_vector_empty(unary_vector_func_fixture):
    arr = pa.array([1], pa.float64())
    result = pc.call_function('y=pct_rank(x)', [arr])
    expected = unary_vector_func_fixture[0](None, arr)
    assert result == expected