import collections
import pprint
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager import record
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
def _wrap_forward_function_with_jvps(self, forward_function, backward_function, inference_args, input_tangents):
    """Adds inline JVP computation to a forward function."""
    forward_wrapper_graph = func_graph_module.FuncGraph(_forward_name(self._func_graph.name))
    with forward_wrapper_graph.as_default():
        with forwardprop_util.push_forwardprop_state():
            forward_captures = {ops.tensor_id(internal): external for external, internal in self._func_graph.captures}
            for input_index, real_input in enumerate(self._func_graph.inputs):
                input_placeholder = array_ops.placeholder(dtype=real_input.dtype, shape=real_input.shape)
                capture = forward_captures.get(ops.tensor_id(real_input))
                if capture is not None:
                    forward_wrapper_graph.add_capture(capture, input_placeholder)
                    if capture.dtype == dtypes.resource:
                        handle_data_util.copy_handle_data(capture, input_placeholder)
                else:
                    forward_wrapper_graph.inputs.append(input_placeholder)
            for inp, arg in zip(forward_wrapper_graph.inputs, inference_args):
                record.record_operation('captured_value', [inp], [arg], backward_function=lambda x: [x], forward_function=lambda x: [x])
            num_inference_inputs = len(inference_args)
            for tape_indices in self._forwardprop_input_indices:
                for input_index, jvp_index in tape_indices:
                    input_placeholder = forward_wrapper_graph.inputs[input_index]
                    if len(forward_wrapper_graph.inputs) != jvp_index:
                        raise errors.InternalError(f'Expected {jvp_index} forward graph inputs, got {len(forward_wrapper_graph.inputs)}.')
                    gradient_shape, gradient_dtype = default_gradient.shape_and_dtype(input_placeholder)
                    jvp_placeholder = graph_placeholder(gradient_dtype, gradient_shape)
                    external_jvp = input_tangents[jvp_index - num_inference_inputs]
                    forward_wrapper_graph.add_capture(external_jvp, jvp_placeholder)
                    tensor_shape.TensorShape(external_jvp.shape).assert_is_compatible_with(jvp_placeholder.shape)
                    record.record_operation('captured_value', [jvp_placeholder], [external_jvp], backward_function=lambda x: [x], forward_function=lambda x: [x])
            forward_inputs = forward_wrapper_graph.inputs[:num_inference_inputs]
            gradient_function = self._delayed_rewrite_functions._rewrite_forward_and_call_backward
            with ops.get_default_graph()._override_gradient_function({'PartitionedCall': gradient_function, 'StatefulPartitionedCall': gradient_function}):
                forward_outputs = forward_function(*forward_inputs)
                if isinstance(forward_outputs, ops.Operation):
                    forward_outputs = []
            py_backward, _ = self._wrap_backward_function(self._func_graph, backward_function, forward_outputs)
        record.record_operation_forwardprop_only(forward_function.cached_definition.signature.name, forward_outputs, forward_inputs, py_backward, None)
        output_indices, output_tangents = pywrap_tfe.TFE_Py_PackJVPs(forward_outputs)
        output_tangents = [forward_wrapper_graph.capture(t) for t in output_tangents]
    return _ForwardWrapper(graph=forward_wrapper_graph, outputs=forward_outputs, output_indices=output_indices, output_tangents=output_tangents)