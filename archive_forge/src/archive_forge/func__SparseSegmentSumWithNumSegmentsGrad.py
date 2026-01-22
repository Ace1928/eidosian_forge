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
@ops.RegisterGradient('SparseSegmentSumWithNumSegments')
def _SparseSegmentSumWithNumSegmentsGrad(op, grad):
    """Gradient for SparseSegmentSumWithNumSegments."""
    dim0 = array_ops.shape(op.inputs[0])[0]
    if compat.forward_compatible(2021, 6, 10):
        return (math_ops.sparse_segment_sum_grad(grad, op.inputs[1], op.inputs[2], dim0), None, None, None)
    else:
        return (math_ops.unsorted_segment_sum(array_ops.gather(grad, op.inputs[2]), op.inputs[1], dim0), None, None, None)