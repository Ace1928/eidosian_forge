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
@ops.RegisterGradient('StatelessIf')
@ops.RegisterGradient('If')
def _IfGrad(op, *grads):
    """The gradient of an If op produced by cond_v2."""
    if_op = op.outputs[0].op
    true_graph, false_graph = get_func_graphs(if_op)
    assert true_graph.outer_graph == if_op.graph
    assert false_graph.outer_graph == if_op.graph
    true_grad_graph = _create_grad_func(true_graph, grads, util.unique_grad_fn_name(true_graph.name))
    false_grad_graph = _create_grad_func(false_graph, grads, util.unique_grad_fn_name(false_graph.name))
    _create_zeros_for_none_grads([true_graph, false_graph], [true_grad_graph, false_grad_graph])
    if true_grad_graph.op_needs_rewrite or false_grad_graph.op_needs_rewrite:
        if control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()):
            true_intermediates = true_grad_graph.xla_intermediates
            false_intermediates = false_grad_graph.xla_intermediates
            extra_true_outputs, extra_false_outputs = _make_intermediates_match_xla([true_graph, false_graph], [true_intermediates, false_intermediates])
        else:
            true_intermediates = true_grad_graph.wrapped_intermediates
            false_intermediates = false_grad_graph.wrapped_intermediates
            extra_true_outputs, extra_false_outputs = _make_intermediates_match([true_graph, false_graph], [true_intermediates, false_intermediates])
        true_graph.outputs.extend(extra_true_outputs)
        false_graph.outputs.extend(extra_false_outputs)
        _check_same_outputs(_COND, [true_graph, false_graph])
        true_graph.name += '_rewritten'
        false_graph.name += '_rewritten'
        if_op._set_func_attr('then_branch', util.create_new_tf_function(true_graph))
        if_op._set_func_attr('else_branch', util.create_new_tf_function(false_graph))
        if_op._set_type_list_attr('Tout', true_graph.output_types)
        if_op._set_shape_list_attr('output_shapes', true_graph.output_shapes)
        if_op._add_outputs([t.dtype for t in extra_true_outputs], [t.shape for t in extra_true_outputs])
    true_grad_inputs = _resolve_grad_inputs(true_graph, true_grad_graph)
    false_grad_inputs = _resolve_grad_inputs(false_graph, false_grad_graph)
    _make_output_composite_tensors_match(_COND, [true_grad_graph, false_grad_graph])
    outputs = _build_cond(if_op.inputs[0], true_grad_graph, false_grad_graph, true_grad_inputs, false_grad_inputs, building_gradient=True)
    return [None] + outputs