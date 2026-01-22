import collections
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
def _py_list_pop(list_, i):
    """Overload of list_pop that executes a Python list append."""
    if i is None:
        x = list_.pop()
    else:
        x = list_.pop(i)
    return (list_, x)