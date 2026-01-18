import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_scalar_agg_empty(unary_agg_func_fixture):
    empty = pa.array([], pa.float64())
    with pytest.raises(pa.ArrowInvalid, match='empty inputs'):
        pc.call_function('mean_udf', [empty])