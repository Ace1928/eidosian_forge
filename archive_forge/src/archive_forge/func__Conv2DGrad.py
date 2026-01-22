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
@ops.RegisterGradient('Conv2D')
def _Conv2DGrad(op, grad):
    """Gradient function for Conv2D."""
    dilations = op.get_attr('dilations')
    strides = op.get_attr('strides')
    padding = op.get_attr('padding')
    explicit_paddings = op.get_attr('explicit_paddings')
    use_cudnn_on_gpu = op.get_attr('use_cudnn_on_gpu')
    data_format = op.get_attr('data_format')
    shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])
    return [gen_nn_ops.conv2d_backprop_input(shape_0, op.inputs[1], grad, dilations=dilations, strides=strides, padding=padding, explicit_paddings=explicit_paddings, use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format), gen_nn_ops.conv2d_backprop_filter(op.inputs[0], shape_1, grad, dilations=dilations, strides=strides, padding=padding, explicit_paddings=explicit_paddings, use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format)]