import collections
from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def _build_cond(pred, true_graph, false_graph, true_inputs, false_inputs, building_gradient, name=None):
    """Creates an If op from the specified predicate, branch functions and inputs.

  Note that this modifies true_graph and false_graph to make the inputs match,
  and to output all intermediates values so they're available for the gradient
  computation.

  true_graph and false_graph need not have the same input types, but they must
  have the same output types.

  Args:
    pred: boolean Tensor
    true_graph: FuncGraph
    false_graph: FuncGraph
    true_inputs: a list of Tensors to be passed to true_graph as input.
    false_inputs: a list of Tensors to be passed to false_graph as input.
    building_gradient: Whether this is a gradient If op.
    name: the name for the If op.

  Returns:
    A list of Tensors which are the outputs of the If op. Does not include added
    intermediate outputs.
  """
    _make_indexed_slices_indices_types_match(_COND, [true_graph, false_graph])
    _check_same_outputs(_COND, [true_graph, false_graph])
    cond_inputs = _make_inputs_match([true_graph, false_graph], [true_inputs, false_inputs])
    if not building_gradient and util.output_all_intermediates():
        true_intermediates = _get_intermediates(true_graph)
        false_intermediates = _get_intermediates(false_graph)
        wrapped_true_intermediates = _wrap_intermediates(true_graph, true_intermediates)
        wrapped_false_intermediates = _wrap_intermediates(false_graph, false_intermediates)
        extra_true_outputs, extra_false_outputs = _make_intermediates_match([true_graph, false_graph], [wrapped_true_intermediates, wrapped_false_intermediates])
        true_graph.outputs.extend(extra_true_outputs)
        false_graph.outputs.extend(extra_false_outputs)
        _check_same_outputs(_COND, [true_graph, false_graph])
    with ops.control_dependencies(list(true_graph.function_captures.control) + list(false_graph.function_captures.control)):
        true_stateful_ops = [op for op in true_graph.get_operations() if op._is_stateful]
        false_stateful_ops = [op for op in false_graph.get_operations() if op._is_stateful]
        if true_stateful_ops or false_stateful_ops:
            op_fn = gen_functional_ops._if
        else:
            op_fn = gen_functional_ops.stateless_if

        def _make_op(inputs):
            if_op, tensors = util.get_op_and_outputs(op_fn(pred, inputs, [t.dtype for t in true_graph.outputs], util.create_new_tf_function(true_graph), util.create_new_tf_function(false_graph), output_shapes=_get_output_shapes(true_graph.outputs, false_graph.outputs), name=name))
            _copy_handle_data(tensors, true_graph.outputs, false_graph.outputs)
            if if_op is not None:
                true_graph.outer_graph = ops.get_default_graph()
                false_graph.outer_graph = ops.get_default_graph()
                if_op._true_graph = true_graph
                if_op._false_graph = false_graph
                util.maybe_set_lowering_attr(if_op)
                util.maybe_propagate_compile_time_consts_in_xla(if_op)
                _set_read_only_resource_inputs_attr(if_op, [true_graph, false_graph])
                if_op.graph.prevent_fetching(if_op)
            return tensors
        tensors = util.run_as_function_for_tape_gradients(_make_op, cond_inputs)
    tensors = [array_ops.identity(t) for t in tensors]
    structured_output_specs = _get_compatible_structured_output_specs(true_graph, false_graph)
    return _pack_sequence_as(structured_output_specs, tensors)