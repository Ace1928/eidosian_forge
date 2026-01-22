from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
@ops.RegisterGradient('SparseReduceSum')
def _SparseReduceSumGrad(op, out_grad):
    """Similar to gradient for the Sum Op (i.e. tf.reduce_sum())."""
    sp_indices = op.inputs[0]
    sp_shape = op.inputs[2]
    output_shape_kept_dims = math_ops.reduced_shape(sp_shape, op.inputs[3])
    out_grad_reshaped = array_ops.reshape(out_grad, output_shape_kept_dims)
    scale = sp_shape // math_ops.cast(output_shape_kept_dims, dtypes.int64)
    return (None, array_ops.gather_nd(out_grad_reshaped, sp_indices // scale), None, None)