import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.fixture(scope='session')
def unary_vector_func_fixture():
    """
    Register a vector function
    """

    def pct_rank(ctx, x):
        return pa.array(x.to_pandas().copy().rank(pct=True))
    func_name = 'y=pct_rank(x)'
    doc = empty_udf_doc
    pc.register_vector_function(pct_rank, func_name, doc, {'x': pa.float64()}, pa.float64())
    return (pct_rank, func_name)