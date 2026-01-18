import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_wrong_input_type_declaration():

    def identity(ctx, val):
        return val
    func_name = 'test_wrong_input_type_declaration'
    in_types = {'array': None}
    out_type = pa.int64()
    doc = {'summary': 'test invalid input type', 'description': 'invalid input function'}
    with pytest.raises(TypeError, match="DataType expected, got <class 'NoneType'>"):
        pc.register_scalar_function(identity, func_name, doc, in_types, out_type)