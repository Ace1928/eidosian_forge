import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_scalar_agg_wrong_output_type(wrong_output_type_agg_func_fixture):
    arr = pa.array([10, 20, 30, 40, 50], pa.int64())
    with pytest.raises(pa.ArrowTypeError, match='output type'):
        pc.call_function('y=wrong_output_type(x)', [arr])