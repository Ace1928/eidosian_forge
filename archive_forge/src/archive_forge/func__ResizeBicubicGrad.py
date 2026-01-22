from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
@ops.RegisterGradient('ResizeBicubic')
def _ResizeBicubicGrad(op, grad):
    """The derivatives for bicubic resizing.

  Args:
    op: The ResizeBicubic op.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the input.
  """
    allowed_types = [dtypes.float32, dtypes.float64]
    grad0 = None
    if op.inputs[0].dtype in allowed_types:
        grad0 = gen_image_ops.resize_bicubic_grad(grad, op.inputs[0], align_corners=op.get_attr('align_corners'), half_pixel_centers=op.get_attr('half_pixel_centers'))
    return [grad0, None]