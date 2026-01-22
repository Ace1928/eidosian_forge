from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.tools.docs import doc_controls
class MaxNorm(Constraint):
    """MaxNorm weight constraint.

  Constrains the weights incident to each hidden unit
  to have a norm less than or equal to a desired value.

  Also available via the shortcut function `tf.keras.constraints.max_norm`.

  Args:
    max_value: the maximum norm value for the incoming weights.
    axis: integer, axis along which to calculate weight norms.
      For instance, in a `Dense` layer the weight matrix
      has shape `(input_dim, output_dim)`,
      set `axis` to `0` to constrain each weight vector
      of length `(input_dim,)`.
      In a `Conv2D` layer with `data_format="channels_last"`,
      the weight tensor has shape
      `(rows, cols, input_depth, output_depth)`,
      set `axis` to `[0, 1, 2]`
      to constrain the weights of each filter tensor of size
      `(rows, cols, input_depth)`.

  """

    def __init__(self, max_value=2, axis=0):
        self.max_value = max_value
        self.axis = axis

    @doc_controls.do_not_generate_docs
    def __call__(self, w):
        norms = backend.sqrt(math_ops.reduce_sum(math_ops.square(w), axis=self.axis, keepdims=True))
        desired = backend.clip(norms, 0, self.max_value)
        return w * (desired / (backend.epsilon() + norms))

    @doc_controls.do_not_generate_docs
    def get_config(self):
        return {'max_value': self.max_value, 'axis': self.axis}