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
def _BroadcastMul(vec, mat):
    """Multiply after broadcasting vec to match dimensions of mat.

  Args:
    vec: A 1-D tensor of dimension [D0]
    mat: A 2-D tensor of dimension [D0, D1]

  Returns:
    A tensor of dimension [D0, D1], the result of vec * mat
  """
    vec = array_ops.expand_dims(vec, -1)
    return vec * mat