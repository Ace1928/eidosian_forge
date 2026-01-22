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
@ops.RegisterGradient('EluGrad')
def _EluGradGrad(op, grad):
    elu_x = op.inputs[1]
    return (gen_nn_ops.elu_grad(grad, elu_x), array_ops.where(elu_x < 0, grad * op.inputs[0], array_ops.zeros_like(elu_x)))