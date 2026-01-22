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
@ops.RegisterGradient('Slice')
def _SliceGrad(op, grad):
    """Gradient for Slice op."""
    input_vec = op.inputs[0]
    begin_vec = op.inputs[1]
    input_rank = array_ops.rank(input_vec)
    index_dtype = begin_vec.dtype
    slice_size = array_ops.shape(op.outputs[0], out_type=index_dtype)
    if control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()):
        return (gen_xla_ops.xla_dynamic_update_slice(array_ops.zeros_like(input_vec), grad, begin_vec), None, None)
    shape = array_ops_stack.stack([input_rank, 1])
    before_pad = array_ops.reshape(begin_vec, shape)
    after_pad = array_ops.reshape(array_ops.shape(input_vec, out_type=index_dtype) - slice_size - begin_vec, shape)
    paddings = array_ops.concat([before_pad, after_pad], 1)
    return (array_ops.pad(grad, paddings), None, None)