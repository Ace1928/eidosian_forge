from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.ops import cond as tf_cond
def _py_if_exp(cond, if_true, if_false):
    return if_true() if cond else if_false()