import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.fixture(scope='session')
def wrong_output_type_func_fixture():
    """
    Register a scalar function which returns something that is neither
    a Arrow scalar or array.
    """

    def wrong_output_type(ctx):
        return 42
    func_name = 'test_wrong_output_type'
    in_types = {}
    out_type = pa.int64()
    doc = {'summary': 'return wrong output type', 'description': ''}
    pc.register_scalar_function(wrong_output_type, func_name, doc, in_types, out_type)
    return (wrong_output_type, func_name)