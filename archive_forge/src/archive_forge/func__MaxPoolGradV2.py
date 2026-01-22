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
@ops.RegisterGradient('MaxPoolV2')
def _MaxPoolGradV2(op, grad):
    ksize = op.inputs[1]
    strides = op.inputs[2]
    return (gen_nn_ops.max_pool_grad_v2(op.inputs[0], op.outputs[0], grad, ksize, strides, padding=op.get_attr('padding'), data_format=op.get_attr('data_format')), None, None)