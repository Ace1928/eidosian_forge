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
@ops.RegisterGradient('Conv3DBackpropFilterV2')
def _Conv3DBackpropFilterGrad(op, grad):
    data_format = op.get_attr('data_format').decode()
    return [nn_ops.conv3d_backprop_input_v2(array_ops.shape(op.inputs[0]), grad, op.inputs[2], dilations=op.get_attr('dilations'), strides=op.get_attr('strides'), padding=op.get_attr('padding'), data_format=data_format), None, nn_ops.conv3d(op.inputs[0], grad, dilations=op.get_attr('dilations'), strides=op.get_attr('strides'), padding=op.get_attr('padding'), data_format=data_format)]