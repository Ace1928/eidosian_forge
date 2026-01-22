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
@classmethod
def _batch_accumulator(cls, primals, tangents):
    """Factory constructor to test accumulator on batches of tangents.

    Args:
      primals: A tensor or nested structure of tensors to watch.
      tangents: A tensor or nested structure of tensors, with the same nesting
        structure as `primals`, with each element being a vector with compatible
        shape `[None] + primal.shape` of the corresponding primal element.

    Returns:
      A batch accumulator object.
    """
    acc = super(ForwardAccumulator, cls).__new__(cls, primals, tangents)
    acc._recording = False
    acc._accumulator = pywrap_tfe.TFE_Py_ForwardAccumulatorNew(True)
    primal_ids = set()
    for primal, tangent in zip(nest.flatten(primals), nest.flatten(tangents)):
        tangent.shape.assert_is_compatible_with(tensor_shape.TensorShape([None]) + primal.shape)
        if id(primal) in primal_ids:
            raise ValueError('Tensor {} was specified as a primal multiple times. This may indicate an error. If it was intended, please sum the corresponding tangents.')
        primal_ids.add(id(primal))
    acc._watch(primals, tangents)
    return acc