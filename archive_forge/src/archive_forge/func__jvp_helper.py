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
def _jvp_helper(op_name, attr_tuple, inputs, outputs, tangents):
    """Computes a Jacobian-vector product for an op.

  Note that this function would be wasteful if executed eagerly. It runs the
  backward gradient function and throws away the result just to record its
  operations on a GradientTape. These unused ops are pruned away when this
  function is traced.

  Args:
    op_name: A string, the type of operation being executed.
    attr_tuple: Attributes of the operation.
    inputs: A flat list of input Tensors to the operation.
    outputs: A flat list of output Tensors from the operation.
    tangents: A flat list of Tensors, same shape as `inputs`.

  Returns:
    A flat list of tangents corresponding to `outputs`.
  """
    with _TRACE_COUNT_CONSISTENCY_LOCK:
        _TRACE_COUNT[op_name] = _TRACE_COUNT.get(op_name, 0) + 1
    special_case = _SPECIAL_CASES.get(op_name, None)
    if special_case is not None:
        return special_case(attr_tuple, inputs, outputs, tangents)
    if not outputs:
        return []
    with forwardprop_util.push_forwardprop_state():
        trainable_inputs = []
        trainable_indices = []
        nontrivial_tangents = []
        for input_index, tensor in enumerate(inputs):
            if backprop_util.IsTrainable(tensor):
                trainable_inputs.append(tensor)
                trainable_indices.append(input_index)
                nontrivial_tangents.append(tangents[input_index])
        with backprop.GradientTape() as transpose_tape:
            with backprop.GradientTape() as backfunc_tape:
                backfunc_tape.watch(trainable_inputs)
                execute.record_gradient(op_name, inputs, attr_tuple, outputs)
            forwardprop_aids = []
            trainable_outputs = []
            nontrivial_output_indices = []
            for output_index, output in enumerate(outputs):
                if backprop_util.IsTrainable(output):
                    forwardprop_aids.append(array_ops.ones_like(output, name='unused_forwardprop_aid'))
                    trainable_outputs.append(output)
                    nontrivial_output_indices.append(output_index)
            transpose_tape.watch(forwardprop_aids)
            grads = backfunc_tape.gradient(trainable_outputs, trainable_inputs, forwardprop_aids, unconnected_gradients=UnconnectedGradients.ZERO)
        nontrivial_output_tangents = transpose_tape.gradient(grads, forwardprop_aids, output_gradients=nontrivial_tangents)
        output_tangents = [None] * len(outputs)
        for index, tangent in zip(nontrivial_output_indices, nontrivial_output_tangents):
            output_tangents[index] = tangent
        return output_tangents