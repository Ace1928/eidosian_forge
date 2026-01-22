import functools
import threading
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import execute
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _jvp_helper_wrapper(op_name, attr_tuple, inputs, outputs, tangents, use_batch):
    """Computes a batch of Jacobian-vector product for an op.

  Args:
    op_name: A string, the type of operation being executed.
    attr_tuple: Attributes of the operation.
    inputs: A flat list of input Tensors to the operation.
    outputs: A flat list of output Tensors from the operation.
    tangents: A flat list of Tensors, compatible with shape `[None] +
      input_shape`.
    use_batch: A bool, True to vetorize over batch of tangents of shape `[None]
      + input_shape`.

  Returns:
    A flat list of tangents compatible with `outputs`
    or `[None] + output_shape`.

  Raises:
    ValueError: if tangent shapes are not compatible with input shapes.
  """
    if use_batch:
        for primal, tangent in zip(inputs, tangents):
            if not tangent.shape.is_compatible_with([None] + primal.shape):
                raise ValueError('Tangent {} was expected to be of shape {} but is instead of shape {}'.format(tangent, [None] + primal.shape, tangent.shape))
        return control_flow_ops.vectorized_map(functools.partial(_jvp_helper, op_name, attr_tuple, inputs, outputs), tangents)
    return _jvp_helper(op_name, attr_tuple, inputs, outputs, tangents)