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
@ops.RegisterGradient('Conv2DBackpropFilter')
def _Conv2DBackpropFilterGrad(op, grad):
    return [gen_nn_ops.conv2d_backprop_input(array_ops.shape(op.inputs[0]), grad, op.inputs[2], dilations=op.get_attr('dilations'), strides=op.get_attr('strides'), padding=op.get_attr('padding'), explicit_paddings=op.get_attr('explicit_paddings'), use_cudnn_on_gpu=op.get_attr('use_cudnn_on_gpu'), data_format=op.get_attr('data_format').decode()), None, gen_nn_ops.conv2d(op.inputs[0], grad, dilations=op.get_attr('dilations'), strides=op.get_attr('strides'), padding=op.get_attr('padding'), explicit_paddings=op.get_attr('explicit_paddings'), use_cudnn_on_gpu=op.get_attr('use_cudnn_on_gpu'), data_format=op.get_attr('data_format').decode())]