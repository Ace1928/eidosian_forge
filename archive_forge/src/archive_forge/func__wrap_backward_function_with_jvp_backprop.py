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
def _wrap_backward_function_with_jvp_backprop(self, backward_function, gradients_wrt_outputs, forward_wrapper):
    """Wraps `backward_function` to include gradients for JVPs."""
    wrapped_backwards_graph = func_graph_module.FuncGraph(_backward_name(self._func_graph.name))
    with wrapped_backwards_graph.as_default():
        py_backward, recorded_outputs = self._wrap_backward_function(self._func_graph, backward_function, forward_wrapper.outputs)
        trainable_index = 0
        forward_doutputs = []
        doutput_args = []
        for output in recorded_outputs:
            if backprop_util.IsTrainable(output):
                doutput = gradients_wrt_outputs[trainable_index]
                doutput_placeholder = graph_placeholder(doutput.dtype, doutput.shape)
                doutput_args.append(doutput_placeholder)
                forward_doutputs.append(doutput_placeholder)
                trainable_index += 1
            else:
                doutput_args.append(None)
        dinputs = py_backward(*doutput_args)
        existing_outputs = object_identity.ObjectIdentitySet(forward_wrapper.outputs + forward_wrapper.output_tangents)
        num_processed_output_tangents = 0
        gradients_wrt_output_tangents = []
        tangent_doutputs = []
        output_tangents = forward_wrapper.output_tangents
        output_indices = forward_wrapper.output_indices
        if self._need_gradients_for_jvps:
            while num_processed_output_tangents != len(output_tangents):
                for output in output_tangents[num_processed_output_tangents:]:
                    gradient_shape, gradient_dtype = default_gradient.shape_and_dtype(output)
                    placeholder = graph_placeholder(gradient_dtype, gradient_shape)
                    gradients_wrt_output_tangents.append(placeholder)
                    tangent_doutputs.append(placeholder)
                num_processed_output_tangents = len(output_tangents)
                with ops.device(None):
                    gradients_wrt_inputs = gradients_util._GradientsHelper(output_tangents, forward_wrapper.graph.inputs, grad_ys=gradients_wrt_output_tangents, src_graph=forward_wrapper.graph)
                dinputs = [backprop_util.AggregateIndexedSlicesGradients((existing, new)) for existing, new in zip(dinputs, gradients_wrt_inputs) if existing is not None or new is not None]
                dinputs.extend(gradients_wrt_inputs[len(dinputs):])
                captures_from_forward = [c for c in wrapped_backwards_graph.external_captures if not isinstance(c, ops.EagerTensor) and c.graph is forward_wrapper.graph]
                for capture in captures_from_forward:
                    if capture not in existing_outputs:
                        existing_outputs.add(capture)
                        forward_wrapper.outputs.append(capture)
                output_indices, output_tangents = forwardprop_util.pack_tangents(forward_wrapper.outputs)
                output_tangents = [forward_wrapper.graph.capture(t) for t in output_tangents]
                for t in output_tangents:
                    existing_outputs.add(t)
    wrapped_backwards_graph.inputs = forward_doutputs[:self._num_trainable_inference_outputs] + tangent_doutputs + forward_doutputs[self._num_trainable_inference_outputs:] + wrapped_backwards_graph.internal_captures
    wrapped_backwards_graph.structured_outputs = dinputs
    wrapped_backwards_graph.outputs = [t for t in dinputs if t is not None]
    return (wrapped_backwards_graph, forward_wrapper._replace(output_indices=output_indices, output_tangents=output_tangents))