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
@ops.RegisterGradient('AvgPool3DGrad')
def _AvgPool3DGradGrad(op, grad):
    return (array_ops.stop_gradient(op.inputs[0]), gen_nn_ops.avg_pool3d(grad, op.get_attr('ksize'), op.get_attr('strides'), op.get_attr('padding'), data_format=op.get_attr('data_format').decode()))