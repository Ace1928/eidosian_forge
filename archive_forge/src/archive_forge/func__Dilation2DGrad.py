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
@ops.RegisterGradient('Dilation2D')
def _Dilation2DGrad(op, grad):
    return [nn_ops.dilation2d_backprop_input(op.inputs[0], op.inputs[1], grad, op.get_attr('strides'), op.get_attr('rates'), op.get_attr('padding')), nn_ops.dilation2d_backprop_filter(op.inputs[0], op.inputs[1], grad, op.get_attr('strides'), op.get_attr('rates'), op.get_attr('padding'))]