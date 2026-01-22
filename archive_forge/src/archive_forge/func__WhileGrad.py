import collections
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util_v1
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_v2_indexed_slices_rewriter
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils
@ops.RegisterGradient('StatelessWhile')
@ops.RegisterGradient('While')
def _WhileGrad(op, *grads):
    """The gradient of a While op produced by while_loop."""
    while_op = op.outputs[0].op
    cond_graph = _get_graph(while_op, 'cond', '_cond_graph')
    body_graph = _get_graph(while_op, 'body', '_body_graph')
    orig_num_params = len(body_graph.outputs)
    maximum_iterations = op.inputs[1]
    parallel_iterations = op.get_attr('parallel_iterations')
    try:
        num_original_outputs = while_op.get_attr('_num_original_outputs')
    except:
        num_original_outputs = len(while_op.outputs)
    num_intermediates = len(while_op.outputs) - num_original_outputs
    grads = [_preprocess_grad(grad, body_out, while_in, while_out) for grad, body_out, while_in, while_out in zip(grads[:num_original_outputs], body_graph.outputs[:num_original_outputs], while_op.inputs[:num_original_outputs], while_op.outputs[:num_original_outputs])] + [None] * num_intermediates
    if getattr(op, 'skip_input_indices', None) is not None:
        captures_start_index = len(body_graph.inputs) - len(body_graph.internal_captures)
        for i in op.skip_input_indices:
            if i >= captures_start_index:
                grads[i] = None
    ys, xs, non_none_grads = zip(*[(y, x, grad) for y, x, grad in zip(body_graph.outputs, body_graph.inputs, grads) if grad is not None])
    body_grad_graph, args = _create_grad_func(ys, xs, non_none_grads, cond_graph, body_graph, util.unique_grad_fn_name(body_graph.name), op, maximum_iterations)
    if body_grad_graph.while_op_needs_rewrite:
        cond_graph.name += '_rewritten'
        body_graph.name += '_rewritten'
        new_inputs = body_grad_graph.extra_inputs
        new_outputs = body_graph.outputs[orig_num_params:]
        while_op._set_func_attr('cond', util.create_new_tf_function(cond_graph))
        while_op._set_func_attr('body', util.create_new_tf_function(body_graph))
        if len(body_graph.output_types) != len(while_op.inputs) + len(new_inputs):
            raise AssertionError(f"Inputs and outputs constructed for the forward op of a While gradient don't match with 'output_types' at  {len(body_graph.output_types)},'inputs' at length {len(while_op.inputs)}, and 'new_inputs' at length {len(new_inputs)}. This doesn't make sense, please file a bug.")
        while_op._set_type_list_attr('T', body_graph.output_types)
        while_op._set_shape_list_attr('output_shapes', body_graph.output_shapes)
        while_op._add_while_inputs(new_inputs)
        while_op._add_outputs([t.dtype for t in new_outputs], [t.shape for t in new_outputs])
        _copy_handle_data(new_outputs, while_op.outputs[orig_num_params:])
    while_op._set_attr('_num_original_outputs', attr_value_pb2.AttrValue(i=len(while_op.outputs)))
    captured_inputs = _resolve_grad_captures(body_graph, body_grad_graph, while_op)
    loop_vars = args + captured_inputs
    loop_vars = while_v2_indexed_slices_rewriter.rewrite_grad_indexed_slices(grads, body_grad_graph, loop_vars, while_op.inputs)

    def grad_cond(counter, unused_maximum_iterations_arg, forward_loop_iters, *unused_args):
        return counter < forward_loop_iters
    grad_cond_name = util.unique_grad_fn_name(op.get_attr('cond').name)
    cond_grad_graph = func_graph_module.func_graph_from_py_func(grad_cond_name, grad_cond, loop_vars, {}, func_graph=util.WhileCondFuncGraph(grad_cond_name))
    _check_num_inputs_outputs(cond_grad_graph, body_grad_graph, len(loop_vars))
    outputs = _build_while_op(loop_vars, cond_grad_graph, body_grad_graph, output_shapes=[t.shape for t in body_grad_graph.outputs], parallel_iterations=parallel_iterations, name='%s_grad' % while_op.name, num_original_outputs=len(body_grad_graph.outputs))
    outputs = [array_ops.identity(t) for t in outputs]
    return _get_structured_grad_output(outputs, grads, body_grad_graph)