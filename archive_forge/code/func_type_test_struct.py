from numba import types
from numba.core import config
@type_callable(TestStruct)
def type_test_struct(context):

    def typer(x, y):
        if isinstance(x, types.Integer) and isinstance(y, types.Integer):
            return test_struct_model_type
    return typer