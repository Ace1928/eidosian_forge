import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_registration_errors():
    doc = {'summary': 'test udf input', 'description': 'parameters are validated'}
    in_types = {'scalar': pa.int64()}
    out_type = pa.int64()

    def test_reg_function(context):
        return pa.array([10])
    with pytest.raises(TypeError):
        pc.register_scalar_function(test_reg_function, None, doc, in_types, out_type)
    with pytest.raises(TypeError, match='func must be a callable'):
        pc.register_scalar_function(None, 'test_none_function', doc, in_types, out_type)
    expected_expr = "DataType expected, got <class 'NoneType'>"
    with pytest.raises(TypeError, match=expected_expr):
        pc.register_scalar_function(test_reg_function, 'test_output_function', doc, in_types, None)
    expected_expr = 'in_types must be a dictionary of DataType'
    with pytest.raises(TypeError, match=expected_expr):
        pc.register_scalar_function(test_reg_function, 'test_input_function', doc, None, out_type)
    pc.register_scalar_function(test_reg_function, 'test_reg_function', doc, {}, out_type)
    expected_expr = 'Already have a function registered with name:' + ' test_reg_function'
    with pytest.raises(KeyError, match=expected_expr):
        pc.register_scalar_function(test_reg_function, 'test_reg_function', doc, {}, out_type)