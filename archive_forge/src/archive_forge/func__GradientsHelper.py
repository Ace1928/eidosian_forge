import collections
import contextlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_state
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _GradientsHelper(ys, xs, grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None, stop_gradients=None, unconnected_gradients=UnconnectedGradients.NONE, src_graph=None):
    """Implementation of gradients()."""
    if context.executing_eagerly():
        raise RuntimeError('tf.gradients is not supported when eager execution is enabled. Use tf.GradientTape instead.')
    ys = variable_utils.convert_variables_to_tensors(_AsList(ys))
    xs = [x.handle if resource_variable_ops.is_resource_variable(x) else x for x in _AsList(xs)]
    if grad_ys is not None:
        grad_ys = _AsList(grad_ys)
    if any((isinstance(x, composite_tensor.CompositeTensor) for x in xs)) or any((isinstance(y, composite_tensor.CompositeTensor) for y in ys)):
        flat_xs = composite_tensor_gradient.get_flat_tensors_for_gradients(xs)
        flat_ys = composite_tensor_gradient.get_flat_tensors_for_gradients(ys)
        flat_grad_ys = None if grad_ys is None else composite_tensor_gradient.get_flat_tensors_for_gradients(grad_ys)
        flat_grads = _GradientsHelper(flat_ys, flat_xs, flat_grad_ys, name, colocate_gradients_with_ops, gate_gradients, aggregation_method, stop_gradients, unconnected_gradients, src_graph)
        return composite_tensor_gradient.replace_flat_tensors_for_gradients(xs, flat_grads)
    if src_graph is None:
        src_graph = ops.get_default_graph()
    try:
        unconnected_gradients = UnconnectedGradients(unconnected_gradients)
    except ValueError:
        raise ValueError(f"Unknown value for unconnected_gradients: '{unconnected_gradients}'")
    func_graphs = []
    curr_graph = src_graph
    while _IsFunction(curr_graph):
        func_graphs.append(curr_graph)
        curr_graph = curr_graph.outer_graph
    stop_gradients = [] if stop_gradients is None else _AsList(stop_gradients)
    if grad_ys is None:
        grad_ys = [None] * len(ys)
    with ops.name_scope(name, 'gradients', list(ys) + list(xs) + list(stop_gradients) + list(grad_ys)) as grad_scope:
        gradient_uid = ops.get_default_graph().unique_name('uid')
        ys = indexed_slices.convert_n_to_tensor_or_indexed_slices(ys, name='y')
        xs = indexed_slices.internal_convert_n_to_tensor_or_indexed_slices(xs, name='x', as_ref=True)
        xs_set = object_identity.ObjectIdentitySet(xs)
        grad_ys = _DefaultGradYs(grad_ys, ys, colocate_gradients_with_ops, gradient_uid)
        to_ops = [t.op for t in ys]
        from_ops = [t.op for t in xs]
        stop_gradient_ops = [t.op for t in stop_gradients]
        reachable_to_ops, pending_count, loop_state = _PendingCount(to_ops, from_ops, colocate_gradients_with_ops, func_graphs, xs_set)
        grads = {}
        for y, grad_y in zip(ys, grad_ys):
            _SetGrad(grads, y, grad_y)
        queue = collections.deque()
        to_ops_set = set()
        for op in to_ops:
            ready = pending_count[op] == 0
            if ready and op not in to_ops_set and (op in reachable_to_ops):
                to_ops_set.add(op)
                queue.append(op)
        if loop_state:
            loop_exits = loop_state.ProcessUnusedLoopExits(pending_count, to_ops_set)
            for y in loop_exits:
                if backprop_util.IsTrainable(y):
                    _SetGrad(grads, y, loop_state.ZerosLikeForExit(y))
                    queue.append(y.op)
        stop_ops = _StopOps(from_ops, stop_gradient_ops, pending_count, xs_set)
        while queue:
            op = queue.popleft()
            with _maybe_colocate_with(op, gradient_uid, colocate_gradients_with_ops):
                if loop_state:
                    loop_state.EnterGradWhileContext(op, before=True)
                out_grads = _AggregatedGrads(grads, op, gradient_uid, loop_state, aggregation_method)
                if loop_state:
                    loop_state.ExitGradWhileContext(op, before=True)
                grad_fn = None
                func_call = None
                is_partitioned_call = _IsPartitionedCall(op)
                is_func_call = src_graph._is_function(op.type) or is_partitioned_call
                has_out_grads = any((isinstance(g, tensor_lib.Tensor) or g for g in out_grads))
                if has_out_grads and op not in stop_ops:
                    try:
                        grad_fn = ops.get_gradient_function(op)
                    except LookupError:
                        if is_func_call:
                            if is_partitioned_call:
                                func_name = compat.as_bytes(op.get_attr('f').name)
                                func_call = src_graph._get_function(func_name)
                                if not func_call and hasattr(src_graph, 'outer_graph'):
                                    graph = src_graph.outer_graph
                                    while graph is not None:
                                        func_call = graph._get_function(func_name)
                                        if func_call is not None:
                                            break
                                        if hasattr(graph, 'outer_graph'):
                                            graph = graph.outer_graph
                                        else:
                                            break
                            else:
                                func_call = src_graph._get_function(op.type)
                            func_call = getattr(op, '__defun', func_call)
                            grad_fn = func_call.python_grad_func
                        else:
                            raise LookupError(f"No gradient defined for operation'{op.name}' (op type: {op.type}). In general every operation must have an associated `@tf.RegisterGradient` for correct autodiff, which this op is lacking. If you want to pretend this operation is a constant in your program, you may insert `tf.stop_gradient`. This can be useful to silence the error in cases where you know gradients are not needed, e.g. the forward pass of tf.custom_gradient. Please see more details in https://www.tensorflow.org/api_docs/python/tf/custom_gradient.")
                if loop_state:
                    loop_state.EnterGradWhileContext(op, before=False)
                if control_flow_util.IsSwitch(op) and op._control_flow_context is not None and op._control_flow_context.IsWhileContext() and (op._control_flow_context == ops.get_default_graph()._get_control_flow_context()):
                    _RaiseNoGradWrtInitialLoopValError(op, from_ops, xs_set)
                if (grad_fn or is_func_call) and has_out_grads:
                    for i, out_grad in enumerate(out_grads):
                        if (not isinstance(out_grad, tensor_lib.Tensor) and (not out_grad)) and (not grad_fn and is_func_call or backprop_util.IsTrainable(op.outputs[i])):
                            if loop_state:
                                out_grads[i] = loop_state.ZerosLikeV1WhileLoop(op, i)
                            elif default_gradient.supports_default_grad(op.outputs[i]):
                                out_grads[i] = control_flow_state.ZerosLike(op, i)
                    with ops.name_scope(op.name + '_grad'):
                        with src_graph._original_op(op):
                            if grad_fn:
                                in_grads = _MaybeCompile(grad_scope, op, func_call, lambda: grad_fn(op, *out_grads))
                            else:
                                in_grads = _MaybeCompile(grad_scope, op, func_call, lambda: _SymGrad(op, out_grads))
                            in_grads = _AsList(in_grads)
                            _VerifyGeneratedGradients(in_grads, op)
                            if gate_gradients and len([x for x in in_grads if x is not None]) > 1:
                                with ops.device(None):
                                    with ops._colocate_with_for_gradient(None, gradient_uid, ignore_existing=True):
                                        in_grads = control_flow_ops.tuple(in_grads)
                    _LogOpGradients(op, out_grads, in_grads)
                else:
                    in_grads = [None] * len(_Inputs(op, xs_set))
                for i, (t_in, in_grad) in enumerate(zip(_Inputs(op, xs_set), in_grads)):
                    if in_grad is not None:
                        if isinstance(in_grad, tensor_lib.Tensor) and t_in.dtype != dtypes.resource:
                            try:
                                in_grad.set_shape(t_in.get_shape())
                            except ValueError:
                                raise ValueError(f'Incompatible shapes between op input and calculated input gradient. Forward operation: {op.name}. Input index: {i}. Original input shape: {t_in.shape}. Calculated input gradient shape: {in_grad.shape}')
                        if not isinstance(t_in, ops.EagerTensor):
                            _SetGrad(grads, t_in, in_grad)
                if loop_state:
                    loop_state.ExitGradWhileContext(op, before=False)
            _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state, xs_set)
    if loop_state:
        loop_state.PostProcessing()
    return [_GetGrad(grads, x, unconnected_gradients) for x in xs]