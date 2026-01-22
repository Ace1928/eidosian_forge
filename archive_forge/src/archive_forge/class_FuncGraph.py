import traceback
from typing import Any, Callable, Hashable
import weakref
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager.polymorphic_function import composite_tensor_utils
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.saved_model import save_context
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.FuncGraph', v1=[])
class FuncGraph(ops.Graph):
    """Graph representing a function body.

  Attributes:
    name: The name of the function.
    inputs: Placeholder tensors representing the inputs to this function. The
      tensors are in this FuncGraph. This represents "regular" inputs as well as
      captured inputs (i.e. the values of self.captures), with the regular
      inputs coming first.
    outputs: Tensors that will be returned by this function. The tensors are in
      this FuncGraph.
    control_outputs: Operations that must be executed before the function
      represented by this graph can be said to have been executed.
    structured_input_signature: A tuple of (args, kwargs), which are both
      possibly-nested python objects that were received by this function. Note
      that these structures might contain Python `None`s.
    structured_outputs: A possibly-nested python object which will be returned
      by this function. The Tensors in this structure are the same as those of
      self.outputs. Note that this structure might contain Python `None`s.
    variables: Variables that should be watched during function execution.
    outer_graph: The graph this function is defined in. May be another FuncGraph
      or the global default Graph.
    captures: Maps external tensor -> internal tensor (i.e. input placeholder).
      The entries are in the order they were captured.
    seed: The graph-level random seed.
    capture_by_value: If True, the func graph will capture Variables by value
      instead of reference.
  """

    def __init__(self, name, collections=None, capture_by_value=None, structured_input_signature=None, structured_outputs=None):
        """Construct a new FuncGraph.

    The graph will inherit its graph key, collections, seed, and distribution
    strategy stack from the current context or graph.

    Args:
      name: the name of the function.
      collections: a dictionary of collections this FuncGraph should start with.
        If not specified (None), the FuncGraph will read (but not write to) the
        outer graph's collections that are not allowlisted, and both read and
        write to the outer graph's collections that are allowlisted. The current
        allowlisted collections are the global variables, the local variables,
        and the trainable variables. Defaults to None.
      capture_by_value: An optional boolean. If True, the func graph will
        capture Variables by value instead of reference. By default inherit from
        outer graphs, and failing that will default to False.
      structured_input_signature: Optional. The structured input signature to
        use for initializing the FuncGraph. See the docstring for FuncGraph for
        more information.
      structured_outputs: Optional. The structured outputs to use for
        initializing the FuncGraph. See the docstring for FuncGraph for more
        information.
    """
        super().__init__()
        self.name = name
        self.inputs = []
        self.outputs = []
        self.control_outputs = []
        self.structured_input_signature = structured_input_signature
        self.structured_outputs = structured_outputs
        self._resource_tensor_inputs = object_identity.ObjectIdentitySet()
        self._weak_variables = []
        self._watched_variables = object_identity.ObjectIdentityWeakSet()
        self.is_control_flow_graph = False
        self._function_captures = capture_container.FunctionCaptures()
        outer_graph = ops.get_default_graph()
        self._weak_outer_graph = weakref.ref(outer_graph)
        while outer_graph.building_function:
            outer_graph = outer_graph.outer_graph
        self._fallback_outer_graph = outer_graph
        self._output_names = None
        if capture_by_value is not None:
            self.capture_by_value = capture_by_value
        elif self.outer_graph is not None and isinstance(self.outer_graph, FuncGraph):
            self.capture_by_value = self.outer_graph.capture_by_value
        else:
            self.capture_by_value = False
        self._building_function = True
        graph = self.outer_graph
        if context.executing_eagerly():
            self.seed = context.global_seed()
            self._seed_used = False
        else:
            self.seed = graph.seed
            self._seed_used = False
            self._colocation_stack = graph._colocation_stack.copy()
        if collections is None:
            for collection_name in graph.get_all_collection_keys():
                if collection_name not in ALLOWLIST_COLLECTIONS:
                    self._collections[collection_name] = graph.get_collection(collection_name)
            for collection_name in ALLOWLIST_COLLECTIONS:
                self._collections[collection_name] = graph.get_collection_ref(collection_name)
        else:
            self._collections = collections
        self._saveable = True
        self._saving_errors = set()
        self._scope_exit_callbacks = None

    def __str__(self):
        return 'FuncGraph(name=%s, id=%s)' % (self.name, id(self))

    def watch_variable(self, v):
        """Marks the variable v as accessed while building this graph."""
        if isinstance(v, resource_variable_ops.ResourceVariable) and v.handle in self._resource_tensor_inputs:
            return
        while self is not None and isinstance(self, FuncGraph):
            self._watched_variables.add(v)
            self = self.outer_graph

    def capture_call_time_value(self, closure, spec, key=None, default_value=None, placeholder=None):
        """Returns a placeholder which at call time has the value closure().

    The `tf.function` supports the notion of captures, that is, it allows Python
    functions to have closure variables, which bind over some value outside the
    function. However, this name binding is "early binding" performed before the
    program is run, i.e.,
    ```
    @tf.function
    def f():
      return x

    x = tf.constant(1)
    f()  # returns 1

    x = tf.constant(2)
    f()  # still returns 1!
    ```
    while in Python, name binding is performed as the program is running.
    ```
    def f():
      return x

    x = 1
    f()  # returns 1

    x = 2
    f()  # returns 2
    ```
    `capture_call_time_value` allows tf.function to mimic late binding as a
    Python function does, by passing in a `closure` callable argument to be
    executed when the tf.function is invoked eagerly.  E.g.
    ```
    @tf.function
    def f():
      return ops.get_default_graph.capture_call_time_value(lambda: x)

    x = tf.constant(1)
    f()  # returns 1

    x = tf.constant(2)
    f()  # returns 2
    ```
    Note that a `capture_call_time_value` function itself does not work well in
    the saving process (since the tf.function in which it's called is not
    invoked eagerly) unless passed a `default_value` argument. At saving time,
    the `default_value` argument is returned instead.

    Args:
      closure: function which takes no arguments, to be evaluated at function
        call time, returning a nest of tensors compatible with `spec`.
      spec: nest of TypeSpec for the value to capture.
      key: optional. If not None, multiple calls to lazy_capture with the same
        key in the same graph will return the same placeholder, and the first
        closure will be used at function call time.
      default_value: optional value to return in environments that cannot safely
        evaluate closure.
      placeholder: optional. If not None, the graph will take the passed-in
        `placeholder` as the internal capture instead of creating a new one.
        This is useful when loading from a SavedModel.

    Returns:
      Nest of placeholders which, at function call time, will be fed with the
      result of calling closure().

    Raises:
      ValueError: at function call time, if the return value of closure() is
       not compatible with `spec`.
    """
        if key is None:
            key = object()
        if key not in self._function_captures.by_ref_internal:
            trace_ctx = trace_type.InternalTracingContext(True)
            spec = trace_type.from_value(spec, trace_ctx)
            if placeholder is None:
                placeholder_ctx = trace_type.InternalPlaceholderContext(self)
                placeholder = spec.placeholder_value(placeholder_ctx)

            def wrapped_closure():
                if save_context.in_save_context() and default_value is not None:
                    return default_value
                if not context.executing_eagerly():
                    graph = ops.get_default_graph()
                    assert isinstance(graph, FuncGraph), 'This API should only be used in TF2 enviroment.'
                    with graph.as_default():
                        ret_nest = graph.capture_call_time_value(closure, spec, key=key, default_value=default_value)
                else:
                    ret_nest = closure()
                ret_nest = spec._cast(ret_nest, trace_type.InternalCastContext)
                return spec._to_tensors(ret_nest)
            wrapped_closure.output_spec = spec
            self._function_captures.add_or_replace(key=key, external=wrapped_closure, internal=placeholder, tracetype=spec, is_by_ref=True)
        return self._function_captures.by_ref_internal[key]

    def control_dependencies(self, control_inputs):
        """Handles control dependencies.

    FuncGraph wraps Graph's control_dependencies logic by first filtering out
    any external tensors / operations and storing them in the graph's
    control_captures member. Any consumers of this function graph must then
    decide how to handle the control captures.

    Args:
      control_inputs: A list of `Operation` or `Tensor` objects which must be
        executed or computed before running the operations defined in the
        context.  Can also be `None` to clear the control dependencies.

    Returns:
     A context manager that specifies control dependencies for all
     operations constructed within the context.

    Raises:
      TypeError: If `control_inputs` is not a list of `Operation` or
        `Tensor` objects.
    """
        if control_inputs is None:
            return super().control_dependencies(control_inputs)
        filtered_control_inputs = []
        for c in control_inputs:
            if isinstance(c, indexed_slices.IndexedSlices) or (hasattr(c, '_handle') and hasattr(c, 'op')):
                c = c.op
            graph_element = ops._as_graph_element(c)
            if graph_element is None:
                graph_element = c
            if graph_element is not None and getattr(graph_element, 'graph', None) is not self:
                self._function_captures.control.add(graph_element)
            else:
                filtered_control_inputs.append(graph_element)
        return super().control_dependencies(filtered_control_inputs)

    def as_default(self):
        outer_cm = super().as_default()

        @tf_contextlib.contextmanager
        def inner_cm():
            """Context manager for copying distribute.Strategy scope information."""
            graph = ops.get_default_graph()
            old_strategy_stack = self._distribution_strategy_stack
            self._distribution_strategy_stack = list(graph._distribution_strategy_stack)
            old_device_stack = self._device_function_stack
            if not context.executing_eagerly() and (device_stack_has_callable(graph._device_function_stack) or (self._distribution_strategy_stack and (not ops.executing_eagerly_outside_functions()))):
                self._device_function_stack = graph._device_function_stack.copy()
            old_creator_stack = self._variable_creator_stack
            self._variable_creator_stack = graph._variable_creator_stack
            old_graph_key = self._graph_key
            self._graph_key = graph._graph_key
            old_scope_exit_callbacks = self._scope_exit_callbacks
            self._scope_exit_callbacks = []
            with outer_cm as g:
                try:
                    yield g
                finally:
                    try:
                        for fn in self._scope_exit_callbacks:
                            fn()
                    finally:
                        self._scope_exit_callbacks = old_scope_exit_callbacks
                        self._distribution_strategy_stack = old_strategy_stack
                        self._device_function_stack = old_device_stack
                        self._variable_creator_stack = old_creator_stack
                        self._graph_key = old_graph_key
        return inner_cm()

    @property
    def outer_graph(self):
        """The Graph this FuncGraph is nested in.

    Functions may capture Tensors from graphs they are nested in (transitive).

    Returns:
      A Graph object. Initially set to the current default graph when the
      FuncGraph was created. If the previous `outer_graph` was deleted because
      the function that owns it was deleted, `outer_graph` is reset to the
      outermost default graph active when the FuncGraph was created. This
      FuncGraph won't have captured anything from the new `outer_graph` (and
      likely not from the previous setting, since that would have created a
      strong reference), but it is returned so that FuncGraphs always have a
      parent.
    """
        current = self._weak_outer_graph()
        if current is None:
            return self._fallback_outer_graph
        return current

    @outer_graph.setter
    def outer_graph(self, new_outer_graph):
        """Sets `outer_graph` to `new_outer_graph`."""
        self._weak_outer_graph = weakref.ref(new_outer_graph)

    @property
    def output_types(self):
        return [t.dtype for t in self.outputs]

    @property
    def output_shapes(self):
        return [t.shape for t in self.outputs]

    @property
    def trainable_variables(self):
        """A sequence of trainable variables accessed by this FuncGraph.

    Note that functions keep only weak references to variables. Calling the
    function after a variable it accesses has been deleted is an error.

    Returns:
      Sequence of trainable variables for this func graph.
    """
        return tuple((v for v in self.variables if v.trainable))

    @property
    def variables(self):
        """A sequence of variables accessed by this FuncGraph.

    Note that functions keep only weak references to variables. Calling the
    function after a variable it accesses has been deleted is an error.

    Returns:
      Sequence of variables for this func graph.
    """

        def deref(weak_v):
            v = weak_v()
            if v is None:
                raise AssertionError('Called a function referencing variables which have been deleted. This likely means that function-local variables were created and not referenced elsewhere in the program. This is generally a mistake; consider storing variables in an object attribute on first call.')
            return v
        return tuple((deref(v) for v in self._weak_variables))

    @variables.setter
    def variables(self, var_list):
        self._weak_variables = [weakref.ref(v) for v in var_list]

    def _capture_by_value(self, op_type, inputs, dtypes, input_types=None, name=None, attrs=None, op_def=None, compute_device=True):
        reverse_captures = dict(((id(v), k) for k, v in self.captures))
        uncaptured_inputs = [reverse_captures.get(id(t), t) for t in inputs]
        with ops.init_scope():
            if context.executing_eagerly():
                attr_list = ('dtype', int(attrs['dtype'].type))
                value, = execute.execute(compat.as_bytes(op_type), 1, uncaptured_inputs, attr_list, context.context())
            else:
                op = ops.get_default_graph()._create_op_internal(op_type, uncaptured_inputs, dtypes, input_types, name, attrs, op_def, compute_device)
                value = op.outputs[0]
        captured_value = self.capture(value)
        return captured_value.op

    def _create_op_internal(self, op_type, inputs, dtypes=None, input_types=None, name=None, attrs=None, op_def=None, compute_device=True):
        """Like Graph.create_op, except handles external input tensors.

    This overload adds functionality to create_op to "capture" any external
    input tensors, i.e. tensors from the eager context or outer function graphs
    if this is a nested function. See `capture` for more information.

    Args:
      op_type: The `Operation` type to create. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
      inputs: A list of `Tensor` objects that will be inputs to the `Operation`.
      dtypes: (Optional) A list of `DType` objects that will be the types of the
        tensors that the operation produces.
      input_types: (Optional.) A list of `DType`s that will be the types of the
        tensors that the operation consumes. By default, uses the base `DType`
        of each input in `inputs`. Operations that expect reference-typed inputs
        must specify `input_types` explicitly.
      name: (Optional.) A string name for the operation. If not specified, a
        name is generated based on `op_type`.
      attrs: (Optional.) A dictionary where the key is the attribute name (a
        string) and the value is the respective `attr` attribute of the
        `NodeDef` proto that will represent the operation (an `AttrValue`
        proto).
      op_def: (Optional.) The `OpDef` proto that describes the `op_type` that
        the operation will have.
      compute_device: (Optional.) If True, device functions will be executed to
        compute the device property of the Operation.

    Returns:
      An `Operation` object.
    """
        if self.capture_by_value and op_type in ['ReadVariableOp', 'ResourceGather']:
            return self._capture_by_value(op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device)
        if op_type == 'Enter' and inputs[0].op.type == 'Enter':
            if inputs[0].op.get_attr('frame_name') == attrs['frame_name'].s:
                return inputs[0].op
        ctxt = ops.get_default_graph()._control_flow_context
        captured_inputs = []
        for inp in inputs:
            if ctxt is not None and hasattr(ctxt, 'AddValue'):
                inp = ctxt.AddValue(inp)
            inp = self.capture(inp)
            captured_inputs.append(inp)
        return super()._create_op_internal(op_type, captured_inputs, dtypes, input_types, name, attrs, op_def, compute_device)

    def capture(self, tensor, name=None, shape=None):
        return self._function_captures.capture_by_value(self, tensor, name)

    def _validate_in_scope(self, tensor):
        inner_graph = tensor.graph
        while inner_graph is not None and isinstance(inner_graph, FuncGraph):
            if inner_graph is self:
                try:
                    tb = tensor.op.traceback
                except AttributeError:
                    tensor_traceback = '<unknown>'
                else:
                    tensor_traceback_list = []
                    for frame in traceback.format_list(tb.get_user_frames()):
                        tensor_traceback_list.extend([f'  {line}' for line in frame.split('\n') if line.strip()])
                    tensor_traceback = '\n'.join(tensor_traceback_list)
                raise errors.InaccessibleTensorError(f'{tensor!r} is out of scope and cannot be used here. Use return values, explicit Python locals or TensorFlow collections to access it.\nPlease see https://www.tensorflow.org/guide/function#all_outputs_of_a_tffunction_must_be_return_values for more information.\n\n{tensor!r} was defined here:\n{tensor_traceback}\n\nThe tensor {tensor!r} cannot be accessed from {self}, because it was defined in {tensor.graph}, which is out of scope.')
            inner_graph = inner_graph.outer_graph

    def _capture_helper(self, tensor, name):
        return self._function_captures._create_placeholder_helper(self, tensor, name)

    def _experimental_capture_side_input_by_ref(self, identifier: Hashable, func: Callable[[], Any]) -> ...:
        """Implement capturing side input by reference for tf.function.

    Note that this API will only register the capture in the func_graph where
    it is called. In the case of nested graph, like nested tf.function or
    tf.while, the outer graph is not aware of this capture in the inner graph.
    Thus, the outer tf.function will not retrace when the by-ref capture
    changes. It's the user's responsibility to call this API in the outer
    func_graph as well if proper retracing is needed.

    For example:

    ```
    x = 1

    # Correct usage
    @tf.function
    def f_1():
      graph = tf.compat.v1.get_default_graph()
      # Capture the same x for the outer tf.function
      graph._experimental_capture_side_input_by_ref("x", lambda: x)

      @tf.function
      def g():
        graph = tf.compat.v1.get_default_graph()
        cap_x = graph._experimental_capture_side_input_by_ref("x", lambda: x)
        return cap_x + 1

      return g()

    # Incorrect usage
    @tf.function
    def f_2():

      @tf.function
      def g():
        graph = tf.compat.v1.get_default_graph()
        cap_x = graph._experimental_capture_side_input_by_ref("x", lambda: x)
        return cap_x + 1

      return g()

    assert f_1() == 2
    assert f_2() == 2
    x = 2
    assert f_1() == 3
    assert f_2() == 2  # This is incorrect
    ```

    Args:
      identifier: A hashable object as the key for the capture.
      func: A Python function that takes no arguments and returns the value of
        side input. The function is evaluated at function call time.

    Returns:
      A nested structure with the same structure as the side input. Tensors
        are replaced with placehoders, and non-tensors remain the same.

    """
        if context.executing_eagerly():
            return func()

        def maybe_convert_to_tensor():
            value = func()
            if not (isinstance(value, core.Value) or isinstance(value, core.Symbol)):
                value = constant_op.constant(value)
            return value
        placeholder = self._function_captures._capture_by_ref(self, maybe_convert_to_tensor, identifier)
        return placeholder

    @property
    def captures(self):
        """Order list of tuples containing external and internal captures."""
        return self._function_captures.by_val_capture_tuples

    def add_capture(self, tensor, placeholder):
        """Capture a specific tensor and utilize the provided placeholder.

    Args:
      tensor: Tensor to captures.
      placeholder: Provided placeholder for the tensor.
    """
        self._function_captures.add_or_replace(key=id(tensor), external=tensor, internal=placeholder, is_by_ref=False)
        self.inputs.append(placeholder)

    def replace_capture(self, tensor, placeholder):
        """Replace already existing capture."""
        self._function_captures.add_or_replace(key=id(tensor), external=tensor, internal=placeholder, is_by_ref=False)

    def replace_capture_with_deferred_capture(self, tensor, closure, spec, placeholder, default_value=None):
        """Replaces existing capture `tensor` with a deferred capture `closure`.

    Caution: It is the caller's responsibility to make sure that, after calling
    this function, the TypeSpec of the `inputs` (i.e. internal placeholders) and
    the `_captured_inputs` (i.e. external captures) of a concrete function that
    wraps this function graph are still compatible. Thus user should pairing
    usage of this function with `ConcreteFunction.set_external_captures` to make
    sure the order still matches. For example,
    ```
    # concrete_fn._captured_inputs == [tensor1, tensor2, tensor3]
    # concrete_fn.inputs == [placeholder1, placeholder2, placeholder3]
    # replace external capture `tensor2` with a deferred_capture, i.e., a
    # closure, `closure2`
    concrete_fn.graph.replace_capture_with_deferred_capture(tensor2,
                                                            closure2,
                                                            placeholder2,
                                                            some_spec,
                                                            some_default)
    concrete_fn.set_external_captures([tensor1, closure2, tensor3])
    ```

    Args:
      tensor: Tensor already captured.
      closure: function which takes no arguments, to be evaluated at function
        call time, returning a nest of tensors compatible with `spec`.
      spec: nest of TypeSpec for the value to capture.
      placeholder: the internal placeholder corresponding to the captured
        `tensor`.
      default_value: optional value to use in environments that cannot safely
        evaluate closure.
    """
        self._function_captures.pop(id(tensor), is_by_ref=False)
        self.capture_call_time_value(closure, spec, key=id(tensor), default_value=default_value, placeholder=placeholder)

    @property
    def external_captures(self):
        """External tensors captured by this function."""
        return list(self._function_captures.by_val_external.values())

    @property
    def internal_captures(self):
        """Placeholders in this function corresponding captured tensors."""
        return list(self._function_captures.by_val_internal.values())

    @property
    def deferred_external_captures(self):
        """Ordered nest of tensors whose placeholders will be fed at call time."""
        return list(self._function_captures.by_ref_external.values())

    @property
    def deferred_internal_captures(self):
        """List of nest of placeholders which at call time will be fed."""
        return list(self._function_captures.by_ref_internal.values())

    @property
    def variable_captures(self):
        """Map of python object ids of variables to variables which are captured."""
        return self.variables

    @property
    def function_captures(self):
        return self._function_captures

    def mark_as_unsaveable(self, error_message):
        """Marks this FuncGraph as unsaveable.

    Any attempts to export this FuncGraph will raise an error with the specified
    message.

    Args:
      error_message: List or string containing the error message to be raised
        when saving this FuncGraph to SavedModel.
    """
        self._saveable = False
        if isinstance(error_message, str):
            error_message = [error_message]
        self._saving_errors.update(error_message)

    @property
    def saveable(self):
        """Returns whether this FuncGraph is saveable."""
        return self._saveable

    @property
    def saving_errors(self):
        """Returns set of errors preventing this FuncGraph from being saved."""
        return self._saving_errors

    def _add_scope_exit_callback(self, fn):
        """Add a function to call when this graph exits the default scope."""
        if not callable(fn):
            raise TypeError('fn is not callable: {}'.format(fn))
        if self._scope_exit_callbacks is None:
            raise RuntimeError("Attempting to add a scope exit callback, but the default graph is not the context scope graph.  Did you forget to call 'with graph.as_default(): ...'?")
        self._scope_exit_callbacks.append(fn)