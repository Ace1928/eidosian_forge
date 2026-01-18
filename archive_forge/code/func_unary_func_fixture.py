import pytest
import pyarrow as pa
from pyarrow import Codec
from pyarrow import fs
import numpy as np
@pytest.fixture(scope='session')
def unary_func_fixture():
    """
    Register a unary scalar function.
    """
    from pyarrow import compute as pc

    def unary_function(ctx, x):
        return pc.call_function('add', [x, 1], memory_pool=ctx.memory_pool)
    func_name = 'y=x+1'
    unary_doc = {'summary': 'add function', 'description': 'test add function'}
    pc.register_scalar_function(unary_function, func_name, unary_doc, {'array': pa.int64()}, pa.int64())
    return (unary_function, func_name)