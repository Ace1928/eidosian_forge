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
@ops.RegisterGradient('Case')
@ops.RegisterGradient('StatelessCase')
def _CaseGrad(op, *grads):
    """The gradient of a Case op produced by tf.switch_case."""
    case_op = op.outputs[0].op
    branch_graphs = get_func_graphs(case_op)
    assert branch_graphs
    for branch_graph in branch_graphs:
        assert branch_graph.outer_graph == case_op.graph
    branch_grad_graphs = []
    for branch_graph in branch_graphs:
        branch_grad_graphs.append(_create_grad_func(branch_graph, grads, util.unique_grad_fn_name(branch_graph.name)))
    _create_zeros_for_none_grads(branch_graphs, branch_grad_graphs)
    if any((g.op_needs_rewrite for g in branch_grad_graphs)):
        if control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()):
            branches_intermediates = [branch_grad_graph.xla_intermediates for branch_grad_graph in branch_grad_graphs]
            extra_branch_outputs = _make_intermediates_match_xla(branch_graphs, branches_intermediates)
        else:
            branch_intermediates = [g.wrapped_intermediates for g in branch_grad_graphs]
            extra_branch_outputs = _make_intermediates_match(branch_graphs, branch_intermediates)
        for branch_graph, extra_outputs in zip(branch_graphs, extra_branch_outputs):
            branch_graph.outputs.extend(extra_outputs)
        _check_same_outputs(_CASE, branch_graphs)
        for branch_graph in branch_graphs:
            branch_graph.name += '_rewritten'
        case_op._set_func_list_attr('branches', [util.create_new_tf_function(branch_graph) for branch_graph in branch_graphs])
        case_op._set_type_list_attr('Tout', branch_graphs[0].output_types)
        case_op._set_shape_list_attr('output_shapes', branch_graphs[0].output_shapes)
        case_op._add_outputs([t.dtype for t in extra_branch_outputs[0]], [t.shape for t in extra_branch_outputs[0]])
    branches_grad_inputs = [_resolve_grad_inputs(branch_graph, branch_grad_graph) for branch_graph, branch_grad_graph in zip(branch_graphs, branch_grad_graphs)]
    _make_output_composite_tensors_match(_CASE, branch_grad_graphs)
    try:
        lowering = case_op._get_attr_bool('_lower_using_switch_merge')
    except errors_impl.NotFoundError:
        lowering = None
    outputs = _build_case(case_op.inputs[0], branch_grad_graphs, branches_grad_inputs, name='gradient', lower_using_switch_merge=lowering)
    return [None] + outputs