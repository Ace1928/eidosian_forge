import functools
import itertools
import operator
from tensorflow.python.eager import backprop
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
@ops.RegisterGradient('Conv2DBackpropInput')
def _Conv2DBackpropInputGrad(op, grad):
    """The derivatives for deconvolution.

  Args:
    op: the Deconvolution op.
    grad: the tensor representing the gradient w.r.t. the output

  Returns:
    the gradients w.r.t. the input and the filter
  """
    return [None, gen_nn_ops.conv2d_backprop_filter(grad, array_ops.shape(op.inputs[1]), op.inputs[2], dilations=op.get_attr('dilations'), strides=op.get_attr('strides'), padding=op.get_attr('padding'), explicit_paddings=op.get_attr('explicit_paddings'), use_cudnn_on_gpu=op.get_attr('use_cudnn_on_gpu'), data_format=op.get_attr('data_format').decode()), gen_nn_ops.conv2d(grad, op.inputs[1], dilations=op.get_attr('dilations'), strides=op.get_attr('strides'), padding=op.get_attr('padding'), explicit_paddings=op.get_attr('explicit_paddings'), use_cudnn_on_gpu=op.get_attr('use_cudnn_on_gpu'), data_format=op.get_attr('data_format').decode())]