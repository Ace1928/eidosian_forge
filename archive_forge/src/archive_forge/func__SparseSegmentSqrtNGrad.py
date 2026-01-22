import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
@ops.RegisterGradient('SparseSegmentSqrtN')
def _SparseSegmentSqrtNGrad(op, grad):
    """Gradient for SparseSegmentSqrtN."""
    dim0 = array_ops.shape(op.inputs[0])[0]
    return (math_ops.sparse_segment_sqrt_n_grad(grad, op.inputs[1], op.inputs[2], dim0), None, None)