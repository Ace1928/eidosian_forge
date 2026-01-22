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
@ops.RegisterGradient('MaxPoolWithArgmax')
def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
    del unused_argmax_grad
    return gen_nn_ops.max_pool_grad_with_argmax(op.inputs[0], grad, op.outputs[1], op.get_attr('ksize'), op.get_attr('strides'), padding=op.get_attr('padding'), include_batch_in_index=op.get_attr('include_batch_in_index'))