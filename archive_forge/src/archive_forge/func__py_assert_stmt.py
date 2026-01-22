from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.util import tf_inspect
def _py_assert_stmt(expression1, expression2):
    """Overload of assert_stmt that executes a Python assert statement."""
    assert expression1, expression2()
    return None