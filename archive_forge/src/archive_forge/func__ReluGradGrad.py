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
@ops.RegisterGradient('ReluGrad')
def _ReluGradGrad(op, grad):
    x = op.inputs[1]
    return (gen_nn_ops.relu_grad(grad, x), array_ops.zeros_like(x))