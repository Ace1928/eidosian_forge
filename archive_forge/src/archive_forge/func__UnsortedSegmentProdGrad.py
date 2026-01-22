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
@ops.RegisterGradient('UnsortedSegmentProd')
def _UnsortedSegmentProdGrad(op, grad):
    """ Gradient for UnsortedSegmentProd.

  The gradient can be expressed for each segment by dividing the segment's
  product by each element of the segment input tensor, but this approach can't
  deal with zeros in the input.
  Unlike reduce_prod we can't use cumsum here as individual segments may have
  a different number of elements. Therefore we consider three cases:
  1) A segment input contains no zeros and we can safely divide by the input
     tensor.
  2) A segment contains exactly one zero. Then the gradient of each input of
     the segment is zero except for the 0-input, there the gradient is
     the product of the remaining segment entries.
  3) A segment contains at least two zeros. The gradient is zero for all
     segment inputs.
  """
    is_zero = math_ops.equal(op.inputs[0], 0)
    num_zeros = gen_math_ops.unsorted_segment_sum(math_ops.cast(is_zero, dtype=dtypes.int32), op.inputs[1], op.inputs[2])
    grad = array_ops.where_v2(math_ops.greater(num_zeros, 1), array_ops.zeros_like(grad), grad)
    non_zero_data = array_ops.where_v2(is_zero, array_ops.ones_like(op.inputs[0]), op.inputs[0])
    non_zero_prod = gen_math_ops.unsorted_segment_prod(non_zero_data, op.inputs[1], op.inputs[2])
    zero_clipped_indices = math_ops.maximum(op.inputs[1], array_ops.zeros_like(op.inputs[1]))
    gathered_prod = array_ops.gather(op.outputs[0], zero_clipped_indices)
    gathered_non_zero_prod = array_ops.gather(non_zero_prod, zero_clipped_indices)
    prod_divided_by_el = gathered_prod / op.inputs[0]
    partial_derivative = array_ops.where_v2(is_zero, gathered_non_zero_prod, prod_divided_by_el)
    gathered_grad = _GatherDropNegatives(grad, op.inputs[1], zero_clipped_indices)[0]
    return (gathered_grad * partial_derivative, None, None)