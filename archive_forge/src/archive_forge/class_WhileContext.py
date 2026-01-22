import abc
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.gen_control_flow_ops import *
from tensorflow.python.util import compat
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
class WhileContext(ControlFlowContext):
    """The context for the loop construct."""

    def __init__(self, maximum_iterations=None, parallel_iterations=10, back_prop=True, swap_memory=False, name='while_context', grad_state=None, context_def=None, import_scope=None):
        """"Creates a `WhileContext`.

    Args:
      maximum_iterations: Optional upper bound on number of loop iterations.
      parallel_iterations: The number of iterations allowed to run in parallel.
      back_prop: Whether backprop is enabled for this while loop.
      swap_memory: Whether GPU-CPU memory swap is enabled for this loop.
      name: Optional name prefix for the returned tensors.
      grad_state: The gradient loop state.
      context_def: Optional `WhileContextDef` protocol buffer to initialize the
        `Whilecontext` python object from.
      import_scope: Optional `string`. Name scope to add. Only used when
        initialing from protocol buffer.
    """
        if context_def:
            self._init_from_proto(context_def, import_scope=import_scope)
        else:
            ControlFlowContext.__init__(self)
            self._init_from_args(maximum_iterations, parallel_iterations, back_prop, swap_memory, name)
        self._grad_state = grad_state

    def _init_from_args(self, maximum_iterations, parallel_iterations, back_prop, swap_memory, name):
        """Creates a new `WhileContext` from arguments.

    Args:
      maximum_iterations: Optional upper bound on number of loop iterations.
      parallel_iterations: The number of iterations allowed to run in parallel.
      back_prop: Whether backprop is enabled for this while loop.
      swap_memory: Whether GPU-CPU memory swap is enabled for this loop.
      name: Optional name prefix for the returned tensors.

    Raises:
      ValueError: If `parallel_iterations` has invalid value.
    """
        if not isinstance(parallel_iterations, int) or parallel_iterations <= 0:
            raise ValueError("'parallel_iterations' must be a positive integer: %s" % parallel_iterations)
        self._name = ops.get_default_graph().unique_name(name)
        self._maximum_iterations = maximum_iterations
        self._parallel_iterations = parallel_iterations
        self._back_prop = back_prop
        self._swap_memory = swap_memory
        self._pivot_for_pred = None
        self._pivot_for_body = None
        self._pivot = None
        self._loop_exits = []
        self._loop_enters = []
        self._graph = ops.get_default_graph()

    def _init_from_proto(self, context_def, import_scope=None):
        """Creates a new `WhileContext` from protocol buffer.

    Args:
      context_def: `WhileContextDef` protocol buffer.
      import_scope: Optional `string`. Name scope to add.
    """
        assert isinstance(context_def, control_flow_pb2.WhileContextDef)
        g = ops.get_default_graph()
        self._name = ops.prepend_name_scope(context_def.context_name, import_scope)
        if context_def.maximum_iterations_name:
            self._maximum_iterations = g.as_graph_element(ops.prepend_name_scope(context_def.maximum_iterations_name, import_scope))
        else:
            self._maximum_iterations = None
        self._parallel_iterations = context_def.parallel_iterations
        self._back_prop = context_def.back_prop
        self._swap_memory = context_def.swap_memory
        self._pivot_for_pred = g.as_graph_element(ops.prepend_name_scope(context_def.pivot_for_pred_name, import_scope))
        self._pivot_for_body = g.as_graph_element(ops.prepend_name_scope(context_def.pivot_for_body_name, import_scope))
        self._pivot = g.as_graph_element(ops.prepend_name_scope(context_def.pivot_name, import_scope))
        self._loop_exits = [g.as_graph_element(ops.prepend_name_scope(exit_name, import_scope)) for exit_name in context_def.loop_exit_names]
        self._loop_enters = [g.as_graph_element(ops.prepend_name_scope(enter_name, import_scope)) for enter_name in context_def.loop_enter_names]
        super(WhileContext, self).__init__(values_def=context_def.values_def, import_scope=import_scope)
        if import_scope:
            for tensor_name in self._values:
                op = g.as_graph_element(tensor_name).op
                if util.IsLoopEnter(op):
                    op._set_attr('frame_name', attr_value_pb2.AttrValue(s=compat.as_bytes(self.name)))
        self._graph = ops.get_default_graph()

    @property
    def maximum_iterations(self):
        """The maximum number of iterations that will be executed."""
        return self._maximum_iterations

    @property
    def parallel_iterations(self):
        """The number of iterations allowed to run in parallel."""
        return self._parallel_iterations

    @property
    def back_prop(self):
        """True iff backprop is enabled for this while loop."""
        return self._back_prop

    @property
    def swap_memory(self):
        """True iff GPU-CPU memory swap is enabled for this while loop."""
        return self._swap_memory

    @property
    def pivot(self):
        """The boolean tensor representing the loop termination condition."""
        return self._pivot

    @property
    def loop_enters(self):
        """The list of enter tensors for loop variables."""
        return self._loop_enters

    @property
    def loop_exits(self):
        """The list of exit tensors for loop variables."""
        return self._loop_exits

    @property
    def grad_state(self):
        """The gradient loop state."""
        return self._grad_state

    def to_proto(self, export_scope=None):
        """Converts a `WhileContext` to a `WhileContextDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `WhileContextDef` protocol buffer.
    """
        if export_scope is None or self.name.startswith(export_scope):
            context_def = control_flow_pb2.WhileContextDef()
            context_def.context_name = ops.strip_name_scope(self.name, export_scope)
            context_def.parallel_iterations = self._parallel_iterations
            if self._maximum_iterations is not None:
                context_def.maximum_iterations_name = ops.strip_name_scope(self._maximum_iterations.name, export_scope)
            context_def.back_prop = self._back_prop
            context_def.swap_memory = self._swap_memory
            context_def.pivot_for_pred_name = ops.strip_name_scope(self._pivot_for_pred.name, export_scope)
            context_def.pivot_for_body_name = ops.strip_name_scope(self._pivot_for_body.name, export_scope)
            context_def.pivot_name = ops.strip_name_scope(self._pivot.name, export_scope)
            context_def.loop_exit_names.extend([ops.strip_name_scope(l.name, export_scope) for l in self._loop_exits])
            context_def.loop_enter_names.extend([ops.strip_name_scope(l.name, export_scope) for l in self._loop_enters])
            context_def.values_def.MergeFrom(super(WhileContext, self)._to_values_def(export_scope=export_scope))
            for nested in self._nested_contexts:
                nested_def = context_def.nested_contexts.add()
                nested.to_control_flow_context_def(nested_def)
            return context_def
        else:
            return None

    def to_control_flow_context_def(self, context_def, export_scope=None):
        context_def.while_ctxt.CopyFrom(self.to_proto(export_scope=export_scope))

    @staticmethod
    def from_proto(context_def, import_scope=None):
        """Returns a `WhileContext` object created from `context_def`.

    Args:
      context_def: A `WhileContextDef` protocol buffer.
      import_scope: Optional `string`. Name scope to add.

    Returns:
      A `WhileContext` Python object.
    """
        ret = WhileContext(context_def=context_def, import_scope=import_scope)
        ret.Enter()
        for nested_def in context_def.nested_contexts:
            from_control_flow_context_def(nested_def, import_scope=import_scope)
        ret.Exit()
        return ret

    def GetWhileContext(self):
        return self

    def GetControlPivot(self):
        if self._pivot_for_body is not None:
            return self._pivot_for_body
        return self._pivot_for_pred

    def AddValue(self, val):
        """Add `val` to the current context and its outer context recursively."""
        result = val
        new_value = val.name not in self._values
        new_value &= val.op._control_flow_context is not self
        if new_value:
            self._values.add(val.name)
            grad_ctxt = ops.get_default_graph()._get_control_flow_context()
            if grad_ctxt:
                grad_ctxt = grad_ctxt.GetWhileContext()
                if grad_ctxt.grad_state:
                    forward_ctxt = util.GetWhileContext(val.op)
                    if util.IsLoopExit(val.op):
                        forward_ctxt = forward_ctxt.outer_context
                        if forward_ctxt:
                            forward_ctxt = forward_ctxt.GetWhileContext()
                    if forward_ctxt == grad_ctxt.grad_state.forward_context:
                        real_val = grad_ctxt.grad_state.GetRealValue(val)
                        self._external_values[val.name] = real_val
                        return real_val
            if self._outer_context is not None:
                result = self._outer_context.AddValue(val)
            with ops.control_dependencies(None):
                enter = _Enter(result, self._name, is_constant=True, parallel_iterations=self._parallel_iterations)
                enter.graph.prevent_feeding(enter)
                if self._outer_context:
                    self._outer_context.AddInnerOp(enter.op)
            self._FixControlInputsAndContext([enter])
            self._values.add(enter.name)
            self._external_values[val.name] = enter
            result = enter
        else:
            actual_val = self._external_values.get(val.name)
            if actual_val is not None:
                result = actual_val
        return result

    def AddOp(self, op):
        """Add `op` to the current context."""
        if not util.IsInXLAContext(op) and op.type in {'Shape', 'Size', 'Rank'}:
            grad_ctxt = ops.get_default_graph()._get_control_flow_context()
            if grad_ctxt:
                grad_ctxt = grad_ctxt.GetWhileContext()
                if grad_ctxt.grad_state:
                    op_input_forward_ctxt = util.GetWhileContext(op.inputs[0].op)
                    if op_input_forward_ctxt == grad_ctxt.grad_state.forward_context:
                        op_input_ctxt = op.inputs[0].op._get_control_flow_context()
                        op._set_control_flow_context(op_input_ctxt)
                        op_input_ctxt._AddOpInternal(op)
                        return
        self._AddOpInternal(op)

    def _AddOpInternal(self, op):
        """Add `op` to the current context.

    We move any external control dependencies of the op to the loop pivot, to
    ensure they get executed.
    """
        if op.type in ['PartitionedCall', 'StatefulPartitionedCall']:
            op._add_control_input(self.GetControlPivot().op)
        if not op.inputs:
            control_inputs, external_inputs = self._RemoveExternalControlEdges(op)
            if not control_inputs:
                op._add_control_input(self.GetControlPivot().op)
            for x in op.outputs:
                self._values.add(x.name)
        else:
            for index in range(len(op.inputs)):
                x = op.inputs[index]
                real_x = self.AddValue(x)
                if real_x != x:
                    op._update_input(index, real_x)
            _, external_inputs = self._RemoveExternalControlEdges(op)
            self._MaybeAddControlDependency(op)
            for x in op.outputs:
                self._values.add(x.name)
        if external_inputs:
            with ops.control_dependencies(None):
                self.Enter()
                external_inputs = [array_ops.identity(x.outputs[0]).op for x in external_inputs if x.outputs]
                self.Exit()
            op._add_control_inputs(external_inputs)
        if self._outer_context or not util.IsLoopExit(op):
            op.graph.prevent_fetching(op)
            for x in op.outputs:
                op.graph.prevent_feeding(x)
        if self._outer_context:
            self._outer_context.AddInnerOp(op)

    def _MaybeAddControlDependency(self, op):
        """Add a control input to the op if it only depends on loop invariants."""

        def _IsOpFree(op):
            """Determines if `op` needs a control dependency."""
            if op.control_inputs:
                return False
            if op.graph._is_function(op.type) or op.type == 'SymbolicGradient':
                return True
            for x in op.inputs:
                if not util.IsLoopConstantEnter(x.op):
                    return False
            return True
        if _IsOpFree(op):
            op._add_control_input(self.GetControlPivot().op)

    def AddForwardLoopCounter(self, outer_grad_state):
        """Adds a loop that counts the number of iterations.

    This is added to the forward loop at the time when we start to
    create the loop for backprop gradient computation. Called in
    the outer context of this forward context.

    The pseudocode is:
      `n = 0; while (_pivot) { n++; }`

    Note that a control dependency is added to `n` to ensure the correct
    execution order of stack push ops.

    Args:
      outer_grad_state: The outer grad state. None if not nested.

    Returns:
      The number of iterations taken by the forward loop and the loop index.
    """
        n = constant_op.constant(0, name='f_count')
        if outer_grad_state is not None:
            outer_add_op = outer_grad_state.forward_index.op.inputs[0].op
            n.op._add_control_input(outer_add_op)
        self.Enter()
        self.AddName(n.name)
        enter_n = _Enter(n, self._name, is_constant=False, parallel_iterations=self._parallel_iterations, name='f_count')
        self.loop_enters.append(enter_n)
        merge_n = merge([enter_n, enter_n])[0]
        switch_n = switch(merge_n, self._pivot)
        index = math_ops.add(switch_n[1], 1)
        next_n = _NextIteration(index)
        merge_n.op._update_input(1, next_n)
        total_iterations = exit(switch_n[0], name='f_count')
        self.loop_exits.append(total_iterations)
        self.ExitResult([total_iterations])
        self.Exit()
        return (total_iterations, next_n)

    def AddBackpropLoopCounter(self, count, outer_grad_state):
        """Add the backprop loop that controls the iterations.

    This is added to the backprop loop. It is used to control the loop
    termination of the backprop loop. Called in the outer context of
    this grad context.

    The pseudocode is:
      `n = count; while (n >= 1) { n--; }`

    Note that a control dependency is added to `final_zero` to ensure the
    correct execution order of stack pop ops.

    Args:
      count: The number of iterations for backprop.
      outer_grad_state: The outer grad state. None if not nested.

    Returns:
      The loop index.
    """
        in_separate_functions = count.graph is not ops.get_default_graph()
        if in_separate_functions:
            count = array_ops.identity(count)
        else:
            one = constant_op.constant(1, name='b_count')
        self.Enter()
        self.AddName(count.name)
        enter_count = _Enter(count, self._name, is_constant=False, parallel_iterations=self._parallel_iterations, name='b_count')
        self.loop_enters.append(enter_count)
        merge_count = merge([enter_count, enter_count])[0]
        self._pivot_for_pred = merge_count
        if in_separate_functions:
            one = constant_op.constant(1, name='b_count')
        pred = math_ops.greater_equal(merge_count, one)
        self._pivot = loop_cond(pred, name='b_count')
        switch_count = switch(merge_count, self._pivot)
        index = math_ops.subtract(switch_count[1], one)
        self._pivot_for_body = index
        next_count = _NextIteration(index)
        merge_count.op._update_input(1, next_count)
        final_zero = exit(switch_count[0], name='b_count')
        self.loop_exits.append(final_zero)
        if outer_grad_state is not None:
            outer_grad_state.grad_sync._add_control_input(final_zero.op)
        self.ExitResult([final_zero])
        self.Exit()
        return next_count

    def AddBackpropAccumulator(self, op, grad):
        """Add an accumulation loop for every loop invariant.

    This is added to the backprop loop. It is used to accumulate partial
    gradients within each loop iteration. Called when in the gradient while
    context.

    The pseudocode is:
      ```
      acc = 0.0;
      while (_pivot) {
        acc += grad;
      }
      ```

    Args:
      op: The Enter op for a loop invariant.
      grad: The partial gradient of an iteration for a loop invariant.

    Returns:
      The gradient for a loop invariant.
    """
        self.Exit()
        shape = grad.get_shape()
        if shape.is_fully_defined():
            if self.outer_context:
                self.outer_context.Enter()
            acc = constant_op.constant(0, grad.dtype, shape=shape, name='b_acc')
            if self.outer_context:
                self.outer_context.Exit()
        else:
            value = op.inputs[0]
            if isinstance(self.outer_context, WhileContext) and self.outer_context.grad_state is not None:
                forward_ctxt = self.grad_state.forward_context
                forward_ctxt.outer_context.Enter()
                zeros_shape = array_ops.shape_internal(value, optimize=False)
                forward_ctxt.outer_context.Exit()
                outer_grad_state = self.grad_state.outer_grad_state
                history_zeros_shape = outer_grad_state.AddForwardAccumulator(zeros_shape)
                self.outer_context.Enter()
                real_shape = outer_grad_state.AddBackpropAccumulatedValue(history_zeros_shape, zeros_shape)
                acc = array_ops.zeros(real_shape, grad.dtype)
                self.outer_context.Exit()
            else:
                if self.outer_context:
                    self.outer_context.Enter()
                zeros_shape = array_ops.shape_internal(value, optimize=False)
                acc = array_ops.zeros(zeros_shape, grad.dtype)
                if self.outer_context:
                    self.outer_context.Exit()
        self.Enter()
        self.AddName(acc.name)
        enter_acc = _Enter(acc, self._name, is_constant=False, parallel_iterations=self._parallel_iterations, name='b_acc')
        self.loop_enters.append(enter_acc)
        merge_acc = merge([enter_acc, enter_acc], name='b_acc')[0]
        switch_acc_false, switch_acc_true = switch(merge_acc, self._pivot)
        add_acc = math_ops.add(switch_acc_true, grad)
        next_acc = _NextIteration(add_acc)
        merge_acc.op._update_input(1, next_acc)
        result_acc = exit(switch_acc_false, name='b_acc')
        self.loop_exits.append(result_acc)
        self.ExitResult([result_acc])
        return result_acc

    def AddBackpropIndexedSlicesAccumulator(self, op, grad):
        """This is used for accumulating gradients that are IndexedSlices.

    This is essentially the equivalent of AddBackpropAccumulator but optimized
    for things like updating embeddings from within a while loop.

    Args:
      op: The Enter op for a loop invariant.
      grad: The partial gradients represented as an IndexedSlices.

    Returns:
      The accumulated IndexedSlices gradient of the loop invariant.
    """
        values = grad.values
        indices = grad.indices
        dense_shape = grad.dense_shape
        self.Exit()
        if self.outer_context:
            self.outer_context.Enter()
        if values.get_shape().is_fully_defined():
            values_shape = tensor_shape.TensorShape([tensor_shape.Dimension(1)] + values.get_shape().dims[1:])
            if self.outer_context:
                self.outer_context.Enter()
            values_acc = constant_op.constant(0, values.dtype, shape=values_shape, name='b_acc')
            if self.outer_context:
                self.outer_context.Exit()
        else:
            values_shape = _resource_safe_shape(op.inputs[0])[1:]
            values_shape = array_ops.concat([[1], values_shape], 0)
            values_acc = array_ops.zeros(values_shape, dtype=values.dtype)
        indices_acc = constant_op.constant([0], indices.dtype)
        shape_acc = None
        if dense_shape is not None:
            if dense_shape.get_shape().is_fully_defined():
                if self.outer_context:
                    self.outer_context.Enter()
                shape_acc = constant_op.constant(0, dense_shape.dtype, shape=dense_shape.get_shape())
                if self.outer_context:
                    self.outer_context.Exit()
            else:
                shape_acc = array_ops.zeros_like(array_ops.shape_internal(op.inputs[0], optimize=False, out_type=dense_shape.dtype), optimize=False)
        if self.outer_context:
            self.outer_context.Exit()
        self.Enter()
        self.AddName(values_acc.name)
        self.AddName(indices_acc.name)
        init_acc = [indices_acc, values_acc]
        if shape_acc is not None:
            self.AddName(shape_acc.name)
            init_acc.append(shape_acc)
        enter_acc = [_Enter(x, self._name, is_constant=False, parallel_iterations=self._parallel_iterations, use_input_shape=False, name='b_acc') for x in init_acc]
        enter_acc[0].set_shape([None])
        if values_acc.shape.dims is not None:
            enter_acc[1].set_shape([None] + values_acc.shape.as_list()[1:])
        self.loop_enters.extend(enter_acc)
        merge_acc = [merge([x, x], name='b_acc')[0] for x in enter_acc]
        switch_acc = [switch(x, self._pivot) for x in merge_acc]
        acc_indexed_slices = [array_ops.concat([xa[1], xv], 0) for xa, xv in zip(switch_acc[:2], [indices, values])]
        if shape_acc is not None:
            acc_indexed_slices.append(math_ops.maximum(dense_shape, switch_acc[2][1]))
        next_acc = [_NextIteration(x) for x in acc_indexed_slices]
        for xm, xn in zip(merge_acc, next_acc):
            xm.op._update_input(1, xn)
        exit_acc = [exit(x[0], name='b_acc') for x in switch_acc]
        self.loop_exits.extend(exit_acc)
        self.ExitResult(exit_acc)
        return indexed_slices.IndexedSlices(indices=exit_acc[0], values=exit_acc[1], dense_shape=exit_acc[2] if shape_acc is not None else None)

    def _InitializeValues(self, values):
        """Makes the values known to this context."""
        self._values = set()
        for x in values:
            if isinstance(x, tensor_lib.Tensor):
                self._values.add(x.name)
            else:
                raise TypeError(f"'values' must be a list of Tensors. Received: {type(x)}.")

    def _BuildLoop(self, pred, body, flat_orig_loop_vars, flat_loop_vars, loop_vars_signature):
        """Core: Add the loop termination condition and body to the graph."""
        flat_shape_invariants = nest.map_structure(lambda spec: spec.shape, nest.flatten(loop_vars_signature, expand_composites=True))
        self._InitializeValues(flat_loop_vars)
        if self._outer_context:
            real_vars = [self._outer_context.AddValue(x) for x in flat_loop_vars]
        else:
            real_vars = flat_loop_vars
        enter_vars = []
        with ops.control_dependencies(None):
            for real_var, shape_invariant in zip(real_vars, flat_shape_invariants):
                enter_var = _Enter(real_var, self._name, is_constant=False, parallel_iterations=self._parallel_iterations, use_input_shape=False)
                if _ShapeLessThanOrEqual(real_var.get_shape(), shape_invariant):
                    enter_var.set_shape(shape_invariant)
                else:
                    raise ValueError(f'The shape invariant specified for {real_var.name} is not compatible with the initial shape of the loop variable. It enters the loop with shape {real_var.get_shape()}, but the specified shape invariant is {shape_invariant}.')
                enter_var.graph.prevent_feeding(enter_var)
                if self._outer_context:
                    self._outer_context.AddInnerOp(enter_var.op)
                enter_vars.append(enter_var)
        outer_context = self._outer_context
        control_pivot = None
        while outer_context is not None and control_pivot is None:
            control_pivot = outer_context.GetControlPivot()
            outer_context = outer_context._outer_context
        if control_pivot is not None:
            for var in enter_vars:
                if util.IsLoopConstantEnter(var.op.inputs[0].op):
                    var.op._add_control_input(control_pivot.op)
        self._FixControlInputsAndContext(enter_vars)
        self._InitializeValues(enter_vars)
        self._loop_enters = enter_vars
        merge_vars = [merge([x, x])[0] for x in enter_vars]
        self._pivot_for_pred = merge_vars[0]
        merge_vars_with_tensorarrays = nest.map_structure(_convert_flow_to_tensorarray, flat_orig_loop_vars, merge_vars)
        packed_vars = nest.pack_sequence_as(structure=loop_vars_signature, flat_sequence=merge_vars_with_tensorarrays, expand_composites=True)
        c = ops.convert_to_tensor(pred(*packed_vars))
        self._pivot = loop_cond(c, name='LoopCond')
        switch_vars = [_SwitchRefOrTensor(x, self._pivot) for x in merge_vars]
        vars_for_body = [_Identity(x[1]) for x in switch_vars]
        self._pivot_for_body = vars_for_body[0]
        vars_for_body_with_tensorarrays = nest.map_structure(_convert_flow_to_tensorarray, flat_orig_loop_vars, vars_for_body)
        packed_vars_for_body = nest.pack_sequence_as(structure=loop_vars_signature, flat_sequence=vars_for_body_with_tensorarrays, expand_composites=True)
        pre_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)
        body_result = body(*packed_vars_for_body)
        post_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)
        if not nest.is_nested(body_result):
            body_result = [body_result]
        if len(post_summaries) > len(pre_summaries):
            new_summaries = post_summaries[len(pre_summaries):]
            summary_ref = ops.get_collection_ref(ops.GraphKeys._SUMMARY_COLLECTION)
            summary_ref[:] = pre_summaries
            with ops.control_dependencies(new_summaries):

                def map_fn(x):
                    if isinstance(x, tensor_array_ops.TensorArray):
                        return x
                    return array_ops.identity(x)
                body_result = nest.map_structure(map_fn, body_result, expand_composites=True)
        body_result = variable_utils.convert_variables_to_tensors(body_result)
        nest.assert_same_structure(list(packed_vars_for_body), list(body_result), expand_composites=True)
        original_body_result = body_result
        result = nest.map_structure(_convert_tensorarray_to_flow, nest.flatten(body_result, expand_composites=True), expand_composites=True)
        result = ops.convert_n_to_tensor_or_composite(result)
        if len(merge_vars) != len(result):
            raise ValueError(f"Number of inputs and outputs of 'body' must match 'loop_vars'. Got {len(merge_vars)} for the number of inputs/outputs, and {len(result)} for 'loop_vars'.")
        next_vars = []
        for m, v in zip(merge_vars, result):
            next_vars.append(_AddNextAndBackEdge(m, v))
        exit_vars = [exit(x[0]) for x in switch_vars]
        self._loop_exits = exit_vars
        self.ExitResult(exit_vars)
        return (original_body_result, exit_vars)

    def BuildLoop(self, pred, body, loop_vars, shape_invariants, return_same_structure):
        """Add the loop termination condition and body to the graph."""
        flat_orig_loop_vars = nest.flatten(loop_vars, expand_composites=True)
        loop_vars = nest.map_structure(_convert_to_tensor_or_composite_or_tensorarray, loop_vars)
        flat_loop_vars = nest.map_structure(_convert_tensorarray_to_flow, nest.flatten(loop_vars, expand_composites=True))
        if shape_invariants is not None:
            loop_vars_signature = nest.map_structure(_shape_invariant_to_type_spec, loop_vars, shape_invariants)
        else:
            loop_vars_signature = nest.map_structure(_shape_invariant_to_type_spec, loop_vars)
        try:
            self.Enter()
            with ops.get_default_graph()._mutation_lock():
                original_body_result, exit_vars = self._BuildLoop(pred, body, flat_orig_loop_vars, flat_loop_vars, loop_vars_signature)
        finally:
            self.Exit()
        flat_result = nest.flatten(original_body_result, expand_composites=True)
        exit_vars_with_tensorarrays = nest.map_structure(_convert_flow_to_tensorarray, flat_result, exit_vars)
        packed_exit_vars = nest.pack_sequence_as(structure=original_body_result, flat_sequence=exit_vars_with_tensorarrays, expand_composites=True)
        if return_same_structure:
            return packed_exit_vars
        else:
            return packed_exit_vars[0] if len(exit_vars) == 1 else packed_exit_vars

    def _FixControlInputsAndContext(self, enters):
        graph = ops.get_default_graph()
        for e in enters:
            if isinstance(e, tensor_lib.Tensor):
                xs = [e]
            else:
                raise TypeError(f"'enters' must be a list of Tensors. Received: {type(e)}.")
            for x in xs:
                inp_op = x.op.inputs[0].op
                control_inputs = graph._control_dependencies_for_inputs([inp_op])
                outer_control_inputs = []
                for op in control_inputs:
                    keep_as_control_input = True
                    op_ctxt = util.GetOutputContext(op)
                    outer_ctxt = self.outer_context
                    outer_while_context = None if outer_ctxt is None else outer_ctxt.GetWhileContext()
                    while outer_ctxt != op_ctxt:
                        if outer_ctxt is None or outer_ctxt == outer_while_context:
                            keep_as_control_input = False
                            break
                        outer_ctxt = outer_ctxt.outer_context
                    if keep_as_control_input:
                        outer_control_inputs.append(op)
                x.op._set_control_flow_context(self)
                x.op._add_control_inputs(outer_control_inputs)
                graph._record_op_seen_by_control_dependencies(x.op)

    def IsWhileContext(self):
        return True