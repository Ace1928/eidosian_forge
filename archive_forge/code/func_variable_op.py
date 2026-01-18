from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops.gen_state_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def variable_op(shape, dtype, name='Variable', set_shape=True, container='', shared_name=''):
    """Deprecated. Used variable_op_v2 instead."""
    if not set_shape:
        shape = tensor_shape.unknown_shape()
    ret = gen_state_ops.variable(shape=shape, dtype=dtype, name=name, container=container, shared_name=shared_name)
    if set_shape:
        ret.set_shape(shape)
    return ret