import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.fixture(scope='session')
def wrong_output_datatype_func_fixture():
    """
    Register a scalar function whose actual output DataType doesn't
    match the declared output DataType.
    """

    def wrong_output_datatype(ctx, array):
        return pc.call_function('add', [array, 1])
    func_name = 'test_wrong_output_datatype'
    in_types = {'array': pa.int64()}
    out_type = pa.int16()
    doc = {'summary': 'return wrong output datatype', 'description': ''}
    pc.register_scalar_function(wrong_output_datatype, func_name, doc, in_types, out_type)
    return (wrong_output_datatype, func_name)