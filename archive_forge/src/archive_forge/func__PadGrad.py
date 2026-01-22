from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
def _PadGrad(op, grad):
    """Gradient for Pad."""
    x = op.inputs[0]
    a = op.inputs[1]
    pad_before = array_ops.slice(a, [0, 0], array_ops_stack.stack([array_ops.rank(x), 1]))
    begin = array_ops.reshape(pad_before, [-1])
    sizes = array_ops.shape(x, out_type=begin.dtype)
    x_grad = array_ops.slice(grad, begin, sizes)
    if len(op.inputs) == 3:
        return (x_grad, None, None)
    else:
        return (x_grad, None)