from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import gen_math_ops
def _py_lazy_and(cond, b):
    """Lazy-eval equivalent of "and" in Python."""
    return cond and b()