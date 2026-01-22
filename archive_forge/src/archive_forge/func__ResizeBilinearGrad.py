from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
@ops.RegisterGradient('ResizeBilinear')
def _ResizeBilinearGrad(op, grad):
    """The derivatives for bilinear resizing.

  Args:
    op: The ResizeBilinear op.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the input.
  """
    grad0 = gen_image_ops.resize_bilinear_grad(grad, op.inputs[0], align_corners=op.get_attr('align_corners'), half_pixel_centers=op.get_attr('half_pixel_centers'))
    return [grad0, None]