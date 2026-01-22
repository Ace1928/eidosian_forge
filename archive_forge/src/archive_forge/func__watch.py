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
def _watch(primal, tangent):
    if not primal.dtype.is_floating:
        logging.log_first_n(logging.WARN, 'The dtype of the watched primal must be floating (e.g. tf.float32), got %r', 5, primal.dtype)
    tangent = ops.convert_to_tensor(tangent, dtype=primal.dtype)
    if hasattr(primal, 'handle'):
        primal = ops.convert_to_tensor(primal.handle)
    pywrap_tfe.TFE_Py_ForwardAccumulatorWatch(self._accumulator, primal, tangent)