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
def _ExtractInputShapes(inputs):
    """Extract the shapes of a set of input tensors."""
    if context.executing_eagerly():
        return array_ops.shape_n(inputs)
    sizes = []
    fully_known = True
    for x in inputs:
        input_shape = array_ops.shape(x)
        if not isinstance(input_shape, tensor.Tensor) or input_shape.op.type != 'Const':
            fully_known = False
            break
        sizes.append(input_shape)
    if fully_known:
        return sizes
    else:
        return array_ops.shape_n(inputs)