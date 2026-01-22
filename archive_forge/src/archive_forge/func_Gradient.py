from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.gen_functional_ops import remote_call
from tensorflow.python.ops.gen_functional_ops import symbolic_gradient
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def Gradient(inputs, f, name=None):
    """Computes the gradient function for function f via backpropagation.

  Args:
    inputs: A list of tensors of size N + M.
    f: The function we want to compute the gradient for.  The function 'f' must
      be a numerical function which takes N inputs and produces M outputs. Its
      gradient function 'g', which is  a function taking N + M inputs and
      produces N outputs.  I.e. if we have (y1, y2, ..., yM) = f(x1, x2, ...,
      xN), then, g is (dL/dx1, dL/dx2, ..., dL/dxN) = g(x1, x2, ..., xN, dL/dy1,
      dL/dy2, ..., dL/dyM),  where L is a scalar-value function of (x1, x2, ...,
      xN) (e.g., the loss function). dL/dxi is the partial derivative of L with
      respect to xi.
    name: A name for the operation (optional).

  Returns:
    A list of tensors of size N.
  """
    tlist = [_.type for _ in f.definition.signature.input_arg]
    return symbolic_gradient(input=inputs, Tout=tlist, f=f, name=name)