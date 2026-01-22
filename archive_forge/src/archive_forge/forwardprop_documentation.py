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
Factory constructor to test accumulator on batches of tangents.

    Args:
      primals: A tensor or nested structure of tensors to watch.
      tangents: A tensor or nested structure of tensors, with the same nesting
        structure as `primals`, with each element being a vector with compatible
        shape `[None] + primal.shape` of the corresponding primal element.

    Returns:
      A batch accumulator object.
    