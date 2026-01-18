import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_varargs_function_validation(varargs_func_fixture):
    _, func_name = varargs_func_fixture
    error_msg = "VarArgs function 'z=ax\\+by\\+c' needs at least 2 arguments"
    with pytest.raises(ValueError, match=error_msg):
        pc.call_function(func_name, [42])