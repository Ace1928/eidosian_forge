from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.control_flow_ops import *
@ops.RegisterGradient('NextIteration')
def _NextIterationGrad(_, grad):
    """A forward next_iteration is translated into a backprop identity.

  Note that the backprop next_iteration is added in switch grad.
  """
    return grad