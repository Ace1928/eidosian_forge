import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.fixture(scope='session')
def wrong_output_type_agg_func_fixture(scope='session'):

    def func(ctx, x):
        return len(x)
    func_name = 'y=wrong_output_type(x)'
    func_doc = empty_udf_doc
    pc.register_aggregate_function(func, func_name, func_doc, {'x': pa.int64()}, pa.int64())
    return (func, func_name)