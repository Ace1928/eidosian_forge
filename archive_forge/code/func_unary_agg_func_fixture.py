import pytest
import pyarrow as pa
from pyarrow import Codec
from pyarrow import fs
import numpy as np
@pytest.fixture(scope='session')
def unary_agg_func_fixture():
    """
    Register a unary aggregate function (mean)
    """
    from pyarrow import compute as pc

    def func(ctx, x):
        return pa.scalar(np.nanmean(x))
    func_name = 'mean_udf'
    func_doc = {'summary': 'y=avg(x)', 'description': 'find mean of x'}
    pc.register_aggregate_function(func, func_name, func_doc, {'x': pa.float64()}, pa.float64())
    return (func, func_name)