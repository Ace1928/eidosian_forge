import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.mark.pandas
def test_vector_struct(struct_vector_func_fixture):
    k = pa.array([1, 1, 2, 2], pa.int64())
    v = pa.array([1.0, 2.0, 3.0, 4.0], pa.float64())
    c = pa.array(['v1', 'v2', 'v1', 'v2'])
    result = pc.call_function('y=pivot(x)', [k, v, c])
    expected = struct_vector_func_fixture[0](None, k, v, c)
    assert result == expected