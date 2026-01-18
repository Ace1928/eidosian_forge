import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.fixture(scope='session')
def varargs_func_fixture():
    """
    Register a varargs scalar function with at least two arguments.
    """

    def varargs_function(ctx, first, *values):
        acc = first
        for val in values:
            acc = pc.call_function('add', [acc, val], memory_pool=ctx.memory_pool)
        return acc
    func_name = 'z=ax+by+c'
    varargs_doc = {'summary': 'z=ax+by+c', 'description': 'find z from z = ax + by + c'}
    pc.register_scalar_function(varargs_function, func_name, varargs_doc, {'array1': pa.int64(), 'array2': pa.int64()}, pa.int64())
    return (varargs_function, func_name)