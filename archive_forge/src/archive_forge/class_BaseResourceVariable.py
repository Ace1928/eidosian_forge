import contextlib
import functools
import weakref
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.checkpoint import tensor_callable
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.compat import compat as forward_compat
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager import tape
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_module
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.gen_resource_variable_ops import *
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
class BaseResourceVariable(variables.Variable, core.Tensor):
    """A python variable from an existing handle."""

    def __init__(self, trainable=None, shape=None, dtype=None, handle=None, constraint=None, synchronization=None, aggregation=None, distribute_strategy=None, name=None, unique_id=None, handle_name=None, graph_element=None, initial_value=None, initializer_op=None, is_initialized_op=None, cached_value=None, save_slice_info=None, caching_device=None, in_graph_mode=None, validate_shape=True, **unused_kwargs):
        """Creates a variable from a handle.

    Args:
      trainable: If `True`, GradientTapes automatically watch uses of this
        Variable.
      shape: The variable's shape. This shape can be set to tf.TensorShape(None)
        in order to assign values of different shapes to this variable.
        Otherwise (i.e. if the shape is fully determined), it will trigger run
        time checks to ensure that each assignment is of the same shape.
      dtype: The variable's dtype.
      handle: The variable's handle
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      distribute_strategy: The distribution strategy this variable was created
        under.
      name: The name for this variable.
      unique_id: Internal. Unique ID for this variable's handle.
      handle_name: The name for the variable's handle.
      graph_element: Optional, required only in session.run-mode. Pre-created
        tensor which reads this variable's value.
      initial_value: Optional. Variable's initial value.
      initializer_op: Operation which assigns the variable's initial value.
      is_initialized_op: Pre-created operation to check whether this variable is
        initialized.
      cached_value: Pre-created operation to read this variable in a specific
        device.
      save_slice_info: Metadata for variable partitioning.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      in_graph_mode: whether we are executing in TF1 graph mode. If None, will
        detect within the function. This is to avoid repeated init_scope()
        conetxt entrances which can add up.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
    """
        if in_graph_mode is None:
            with ops.init_scope():
                self._in_graph_mode = not context.executing_eagerly()
        else:
            self._in_graph_mode = in_graph_mode
        synchronization, aggregation, trainable = variables.validate_synchronization_aggregation_trainable(synchronization, aggregation, trainable, name)
        self._trainable = trainable
        self._synchronization = synchronization
        self._aggregation = aggregation
        self._save_slice_info = save_slice_info
        self._initial_value = initial_value
        self._initializer_op = initializer_op
        self._is_initialized_op = is_initialized_op
        self._graph_element = graph_element
        self._caching_device = caching_device
        self._cached_value = cached_value
        self._distribute_strategy = distribute_strategy
        self._graph_key = ops.get_default_graph()._graph_key
        self._shape = tensor_shape.as_shape(shape)
        self._dtype = dtypes.as_dtype(dtype)
        self._handle = handle
        self._unique_id = unique_id
        if handle_name is None:
            self._handle_name = 'Variable:0'
        else:
            self._handle_name = handle_name + ':0'
        self._constraint = constraint
        self._cached_shape_as_list = None
        self._validate_shape = validate_shape

    def __repr__(self):
        if context.executing_eagerly() and (not self._in_graph_mode):
            try:
                with ops.device(self.device):
                    value_text = ops.value_text(self.read_value(), is_repr=True)
            except:
                value_text = 'numpy=<unavailable>'
            return "<tf.Variable '%s' shape=%s dtype=%s, %s>" % (self.name, self.get_shape(), self.dtype.name, value_text)
        else:
            return "<tf.Variable '%s' shape=%s dtype=%s>" % (self.name, self.get_shape(), self.dtype.name)

    def __tf_tracing_type__(self, signature_context):
        alias_id = signature_context.alias_global_id(self._handle._id)
        signature_context.add_placeholder(alias_id, self)
        return VariableSpec(shape=self.shape, dtype=self.dtype, trainable=self.trainable, alias_id=alias_id)

    @contextlib.contextmanager
    def _assign_dependencies(self):
        """Makes assignments depend on the cached value, if any.

    This prevents undefined behavior with reads not ordered wrt writes.

    Yields:
      None.
    """
        if self._cached_value is not None:
            with ops.control_dependencies([self._cached_value]):
                yield
        else:
            yield

    def __array__(self, dtype=None):
        """Allows direct conversion to a numpy array.

    >>> np.array(tf.Variable([1.0]))
    array([1.], dtype=float32)

    Returns:
      The variable value as a numpy array.
    """
        return np.asarray(self.numpy(), dtype=dtype)

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        return bool(self.read_value())

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        if not context.executing_eagerly():
            raise NotImplementedError('__deepcopy__() is only available when eager execution is enabled.')
        copied_variable = ResourceVariable(initial_value=self.read_value(), trainable=self._trainable, constraint=self._constraint, dtype=self._dtype, name=self._shared_name, distribute_strategy=self._distribute_strategy, synchronization=self.synchronization, aggregation=self.aggregation)
        memo[self._unique_id] = copied_variable
        return copied_variable

    @property
    def dtype(self):
        """The dtype of this variable."""
        return self._dtype

    @property
    def device(self):
        """The device this variable is on."""
        return self.handle.device

    @property
    def graph(self):
        """The `Graph` of this variable."""
        return self.handle.graph

    @property
    def name(self):
        """The name of the handle for this variable."""
        return self._handle_name

    @property
    def shape(self):
        """The shape of this variable."""
        return self._shape

    def set_shape(self, shape):
        self._shape = self._shape.merge_with(shape)

    def _shape_as_list(self):
        if self.shape.ndims is None:
            return None
        return [dim.value for dim in self.shape.dims]

    def _shape_tuple(self):
        shape = self._shape_as_list()
        if shape is None:
            return None
        return tuple(shape)

    @property
    def create(self):
        """The op responsible for initializing this variable."""
        if not self._in_graph_mode:
            raise RuntimeError('This operation is not supported when eager execution is enabled.')
        return self._initializer_op

    @property
    def handle(self):
        """The handle by which this variable can be accessed."""
        return self._handle

    def value(self):
        """A cached operation which reads the value of this variable."""
        if self._cached_value is not None:
            return self._cached_value
        with ops.colocate_with(None, ignore_existing=True):
            return self._read_variable_op()

    def _as_graph_element(self):
        """Conversion function for Graph.as_graph_element()."""
        return self._graph_element

    @property
    def initializer(self):
        """The op responsible for initializing this variable."""
        return self._initializer_op

    @property
    def initial_value(self):
        """Returns the Tensor used as the initial value for the variable."""
        if context.executing_eagerly():
            raise RuntimeError('This property is not supported when eager execution is enabled.')
        return self._initial_value

    @property
    def constraint(self):
        """Returns the constraint function associated with this variable.

    Returns:
      The constraint function that was passed to the variable constructor.
      Can be `None` if no constraint was passed.
    """
        return self._constraint

    @property
    def op(self):
        """The op for this variable."""
        return self.handle.op

    @property
    def trainable(self):
        return self._trainable

    @property
    def synchronization(self):
        return self._synchronization

    @property
    def aggregation(self):
        return self._aggregation

    def eval(self, session=None):
        """Evaluates and returns the value of this variable."""
        if context.executing_eagerly():
            raise RuntimeError('This operation is not supported when eager execution is enabled.')
        return self._graph_element.eval(session=session)

    def numpy(self):
        if context.executing_eagerly():
            return self.read_value().numpy()
        raise NotImplementedError('numpy() is only available when eager execution is enabled.')

    @deprecated(None, 'Prefer Dataset.range instead.')
    def count_up_to(self, limit):
        """Increments this variable until it reaches `limit`.

    When that Op is run it tries to increment the variable by `1`. If
    incrementing the variable would bring it above `limit` then the Op raises
    the exception `OutOfRangeError`.

    If no error is raised, the Op outputs the value of the variable before
    the increment.

    This is essentially a shortcut for `count_up_to(self, limit)`.

    Args:
      limit: value at which incrementing the variable raises an error.

    Returns:
      A `Tensor` that will hold the variable value before the increment. If no
      other Op modifies this variable, the values produced will all be
      distinct.
    """
        return gen_state_ops.resource_count_up_to(self.handle, limit=limit, T=self.dtype)

    def _export_to_saved_model_graph(self, object_map=None, tensor_map=None, options=None, **kwargs):
        """For implementing `Trackable`."""
        new_variable = None
        if options.experimental_variable_policy._save_variable_devices():
            with ops.device(self.device):
                new_variable = copy_to_graph_uninitialized(self)
        else:
            new_variable = copy_to_graph_uninitialized(self)
        object_map[self] = new_variable
        tensor_map[self.handle] = new_variable.handle
        return [self.handle]

    def _serialize_to_tensors(self):
        """Implements Trackable._serialize_to_tensors."""

        def _read_variable_closure():
            v = self
            with ops.device(v.device):
                if context.executing_eagerly() and (not v.is_initialized()):
                    return None
                x = v.read_value_no_copy()
                with ops.device('/device:CPU:0'):
                    return array_ops.identity(x)
        return {trackable.VARIABLE_VALUE_KEY: tensor_callable.Callable(_read_variable_closure, dtype=self.dtype, device=self.device)}

    def _restore_from_tensors(self, restored_tensors):
        """Implements Trackable._restore_from_tensors."""
        with ops.device(self.device):
            restored_tensor = array_ops.identity(restored_tensors[trackable.VARIABLE_VALUE_KEY])
            try:
                assigned_variable = shape_safe_assign_variable_handle(self.handle, self.shape, restored_tensor)
            except ValueError as e:
                raise ValueError(f'Received incompatible tensor with shape {restored_tensor.shape} when attempting to restore variable with shape {self.shape} and name {self.name}.') from e
            return assigned_variable

    def _read_variable_op(self, no_copy=False):
        """Reads the value of the variable.

    If the variable is in copy-on-read mode and `no_copy` is True, the variable
    is converted to copy-on-write mode before it is read.

    Args:
      no_copy: Whether to prevent a copy of the variable.

    Returns:
      The value of the variable.
    """
        variable_accessed(self)

        def read_and_set_handle(no_copy):
            if no_copy and forward_compat.forward_compatible(2022, 5, 3):
                gen_resource_variable_ops.disable_copy_on_read(self.handle)
            result = gen_resource_variable_ops.read_variable_op(self.handle, self._dtype)
            _maybe_set_handle_data(self._dtype, self.handle, result)
            return result
        if getattr(self, '_caching_device', None) is not None:
            with ops.colocate_with(None, ignore_existing=True):
                with ops.device(self._caching_device):
                    result = read_and_set_handle(no_copy)
        else:
            result = read_and_set_handle(no_copy)
        if not context.executing_eagerly():
            record.record_operation('ReadVariableOp', [result], [self.handle], backward_function=lambda x: [x], forward_function=lambda x: [x])
        return result

    def read_value(self):
        """Constructs an op which reads the value of this variable.

    Should be used when there are multiple reads, or when it is desirable to
    read the value only after some condition is true.

    Returns:
      The value of the variable.
    """
        with ops.name_scope('Read'):
            value = self._read_variable_op()
        return array_ops.identity(value)

    def read_value_no_copy(self):
        """Constructs an op which reads the value of this variable without copy.

    The variable is read without making a copy even when it has been sparsely
    accessed. Variables in copy-on-read mode will be converted to copy-on-write
    mode.

    Returns:
      The value of the variable.
    """
        with ops.name_scope('Read'):
            value = self._read_variable_op(no_copy=True)
        return array_ops.identity(value)

    def sparse_read(self, indices, name=None):
        """Reads the value of this variable sparsely, using `gather`."""
        with ops.name_scope('Gather' if name is None else name) as name:
            variable_accessed(self)
            value = gen_resource_variable_ops.resource_gather(self.handle, indices, dtype=self._dtype, name=name)
            if self._dtype == dtypes.variant:
                handle_data = get_eager_safe_handle_data(self.handle)
                if handle_data.is_set and len(handle_data.shape_and_type) > 1:
                    value._handle_data = cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData(is_set=True, shape_and_type=handle_data.shape_and_type[1:])
                return array_ops.identity(value)
        return value

    def gather_nd(self, indices, name=None):
        """Reads the value of this variable sparsely, using `gather_nd`."""
        with ops.name_scope('GatherNd' if name is None else name) as name:
            if self.trainable:
                variable_accessed(self)
            value = gen_resource_variable_ops.resource_gather_nd(self.handle, indices, dtype=self._dtype, name=name)
        return array_ops.identity(value)

    def to_proto(self, export_scope=None):
        """Converts a `ResourceVariable` to a `VariableDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Raises:
      RuntimeError: If run in EAGER mode.

    Returns:
      A `VariableDef` protocol buffer, or `None` if the `Variable` is not
      in the specified name scope.
    """
        if context.executing_eagerly():
            raise RuntimeError('This operation is not supported when eager execution is enabled.')
        if export_scope is None or self.handle.name.startswith(export_scope):
            var_def = variable_pb2.VariableDef()
            var_def.variable_name = ops.strip_name_scope(self.handle.name, export_scope)
            if self._initial_value is not None:
                var_def.initial_value_name = ops.strip_name_scope(self._initial_value.name, export_scope)
            var_def.initializer_name = ops.strip_name_scope(self.initializer.name, export_scope)
            if self._cached_value is not None:
                var_def.snapshot_name = ops.strip_name_scope(self._cached_value.name, export_scope)
            else:
                var_def.snapshot_name = ops.strip_name_scope(self._graph_element.name, export_scope)
            var_def.is_resource = True
            var_def.trainable = self.trainable
            var_def.synchronization = self.synchronization.value
            var_def.aggregation = self.aggregation.value
            if self._save_slice_info:
                var_def.save_slice_info_def.MergeFrom(self._save_slice_info.to_proto(export_scope=export_scope))
            return var_def
        else:
            return None

    @staticmethod
    def from_proto(variable_def, import_scope=None):
        if context.executing_eagerly():
            raise RuntimeError('This operation is not supported when eager execution is enabled.')
        return ResourceVariable(variable_def=variable_def, import_scope=import_scope)
    __array_priority__ = 100

    def is_initialized(self, name=None):
        """Checks whether a resource variable has been initialized.

    Outputs boolean scalar indicating whether the tensor has been initialized.

    Args:
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `bool`.
    """
        return gen_resource_variable_ops.var_is_initialized_op(self.handle, name)

    def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
        """Subtracts a value from this variable.

    Args:
      delta: A `Tensor`. The value to subtract from this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name to use for the operation.
      read_value: A `bool`. Whether to read and return the new value of the
        variable or not.

    Returns:
      If `read_value` is `True`, this method will return the new value of the
      variable after the assignment has completed. Otherwise, when in graph mode
      it will return the `Operation` that does the assignment, and when in eager
      mode it will return `None`.
    """
        with _handle_graph(self.handle), self._assign_dependencies():
            assign_sub_op = gen_resource_variable_ops.assign_sub_variable_op(self.handle, ops.convert_to_tensor(delta, dtype=self.dtype), name=name)
        if read_value:
            return self._lazy_read(assign_sub_op)
        return assign_sub_op

    def assign_add(self, delta, use_locking=None, name=None, read_value=True):
        """Adds a value to this variable.

    Args:
      delta: A `Tensor`. The value to add to this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name to use for the operation.
      read_value: A `bool`. Whether to read and return the new value of the
        variable or not.

    Returns:
      If `read_value` is `True`, this method will return the new value of the
      variable after the assignment has completed. Otherwise, when in graph mode
      it will return the `Operation` that does the assignment, and when in eager
      mode it will return `None`.
    """
        with _handle_graph(self.handle), self._assign_dependencies():
            assign_add_op = gen_resource_variable_ops.assign_add_variable_op(self.handle, ops.convert_to_tensor(delta, dtype=self.dtype), name=name)
        if read_value:
            return self._lazy_read(assign_add_op)
        return assign_add_op

    def _lazy_read(self, op):
        variable_accessed(self)
        return _UnreadVariable(handle=self.handle, dtype=self.dtype, shape=self._shape, in_graph_mode=self._in_graph_mode, parent_op=op, unique_id=self._unique_id)

    def assign(self, value, use_locking=None, name=None, read_value=True):
        """Assigns a new value to this variable.

    Args:
      value: A `Tensor`. The new value for this variable.
      use_locking: If `True`, use locking during the assignment.
      name: The name to use for the assignment.
      read_value: A `bool`. Whether to read and return the new value of the
        variable or not.

    Returns:
      If `read_value` is `True`, this method will return the new value of the
      variable after the assignment has completed. Otherwise, when in graph mode
      it will return the `Operation` that does the assignment, and when in eager
      mode it will return `None`.
    """
        with _handle_graph(self.handle):
            value_tensor = ops.convert_to_tensor(value, dtype=self.dtype)
            if not self._shape.is_compatible_with(value_tensor.shape):
                if self.name is None:
                    tensor_name = ''
                else:
                    tensor_name = ' ' + str(self.name)
                raise ValueError(f"Cannot assign value to variable '{tensor_name}': Shape mismatch.The variable shape {self._shape}, and the assigned value shape {value_tensor.shape} are incompatible.")
            kwargs = {}
            if forward_compat.forward_compatible(2022, 3, 23):
                validate_shape = self._validate_shape and self._shape.is_fully_defined()
                kwargs['validate_shape'] = validate_shape
            assign_op = gen_resource_variable_ops.assign_variable_op(self.handle, value_tensor, name=name, **kwargs)
            if read_value:
                return self._lazy_read(assign_op)
        return assign_op

    def __reduce__(self):
        return (functools.partial(ResourceVariable, initial_value=self.numpy(), trainable=self.trainable, name=self._shared_name, dtype=self.dtype, constraint=self.constraint, distribute_strategy=self._distribute_strategy), ())

    def scatter_sub(self, sparse_delta, use_locking=False, name=None):
        """Subtracts `tf.IndexedSlices` from this variable.

    Args:
      sparse_delta: `tf.IndexedSlices` to be subtracted from this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(gen_resource_variable_ops.resource_scatter_sub(self.handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

    def scatter_add(self, sparse_delta, use_locking=False, name=None):
        """Adds `tf.IndexedSlices` to this variable.

    Args:
      sparse_delta: `tf.IndexedSlices` to be added to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(gen_resource_variable_ops.resource_scatter_add(self.handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

    def scatter_max(self, sparse_delta, use_locking=False, name=None):
        """Updates this variable with the max of `tf.IndexedSlices` and itself.

    Args:
      sparse_delta: `tf.IndexedSlices` to use as an argument of max with this
        variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(gen_resource_variable_ops.resource_scatter_max(self.handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

    def scatter_min(self, sparse_delta, use_locking=False, name=None):
        """Updates this variable with the min of `tf.IndexedSlices` and itself.

    Args:
      sparse_delta: `tf.IndexedSlices` to use as an argument of min with this
        variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(gen_resource_variable_ops.resource_scatter_min(self.handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

    def scatter_mul(self, sparse_delta, use_locking=False, name=None):
        """Multiply this variable by `tf.IndexedSlices`.

    Args:
      sparse_delta: `tf.IndexedSlices` to multiply this variable by.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(gen_resource_variable_ops.resource_scatter_mul(self.handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

    def scatter_div(self, sparse_delta, use_locking=False, name=None):
        """Divide this variable by `tf.IndexedSlices`.

    Args:
      sparse_delta: `tf.IndexedSlices` to divide this variable by.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(gen_resource_variable_ops.resource_scatter_div(self.handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

    def scatter_update(self, sparse_delta, use_locking=False, name=None):
        """Assigns `tf.IndexedSlices` to this variable.

    Args:
      sparse_delta: `tf.IndexedSlices` to be assigned to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(gen_resource_variable_ops.resource_scatter_update(self.handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

    def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
        """Assigns `tf.IndexedSlices` to this variable batch-wise.

    Analogous to `batch_gather`. This assumes that this variable and the
    sparse_delta IndexedSlices have a series of leading dimensions that are the
    same for all of them, and the updates are performed on the last dimension of
    indices. In other words, the dimensions should be the following:

    `num_prefix_dims = sparse_delta.indices.ndims - 1`
    `batch_dim = num_prefix_dims + 1`
    `sparse_delta.updates.shape = sparse_delta.indices.shape + var.shape[
         batch_dim:]`

    where

    `sparse_delta.updates.shape[:num_prefix_dims]`
    `== sparse_delta.indices.shape[:num_prefix_dims]`
    `== var.shape[:num_prefix_dims]`

    And the operation performed can be expressed as:

    `var[i_1, ..., i_n,
         sparse_delta.indices[i_1, ..., i_n, j]] = sparse_delta.updates[
            i_1, ..., i_n, j]`

    When sparse_delta.indices is a 1D tensor, this operation is equivalent to
    `scatter_update`.

    To avoid this operation one can looping over the first `ndims` of the
    variable and using `scatter_update` on the subtensors that result of slicing
    the first dimension. This is a valid option for `ndims = 1`, but less
    efficient than this implementation.

    Args:
      sparse_delta: `tf.IndexedSlices` to be assigned to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(state_ops.batch_scatter_update(self, sparse_delta.indices, sparse_delta.values, use_locking=use_locking, name=name))

    def scatter_nd_sub(self, indices, updates, name=None):
        """Applies sparse subtraction to individual values or slices in a Variable.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        op = ref.scatter_nd_sub(indices, updates)
        with tf.compat.v1.Session() as sess:
          print sess.run(op)
    ```

    The resulting update to ref would look like this:

        [1, -9, 3, -6, -6, 6, 7, -4]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      The updated variable.
    """
        return self._lazy_read(gen_state_ops.resource_scatter_nd_sub(self.handle, indices, ops.convert_to_tensor(updates, self.dtype), name=name))

    def scatter_nd_add(self, indices, updates, name=None):
        """Applies sparse addition to individual values or slices in a Variable.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        add = ref.scatter_nd_add(indices, updates)
        with tf.compat.v1.Session() as sess:
          print sess.run(add)
    ```

    The resulting update to ref would look like this:

        [1, 13, 3, 14, 14, 6, 7, 20]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      The updated variable.
    """
        return self._lazy_read(gen_state_ops.resource_scatter_nd_add(self.handle, indices, ops.convert_to_tensor(updates, self.dtype), name=name))

    def scatter_nd_update(self, indices, updates, name=None):
        """Applies sparse assignment to individual values or slices in a Variable.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        op = ref.scatter_nd_update(indices, updates)
        with tf.compat.v1.Session() as sess:
          print sess.run(op)
    ```

    The resulting update to ref would look like this:

        [1, 11, 3, 10, 9, 6, 7, 12]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      The updated variable.
    """
        return self._lazy_read(gen_state_ops.resource_scatter_nd_update(self.handle, indices, ops.convert_to_tensor(updates, self.dtype), name=name))

    def scatter_nd_max(self, indices, updates, name=None):
        """Updates this variable with the max of `tf.IndexedSlices` and itself.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      The updated variable.
    """
        return self._lazy_read(gen_state_ops.resource_scatter_nd_max(self.handle, indices, ops.convert_to_tensor(updates, self.dtype), name=name))

    def scatter_nd_min(self, indices, updates, name=None):
        """Updates this variable with the min of `tf.IndexedSlices` and itself.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      The updated variable.
    """
        return self._lazy_read(gen_state_ops.resource_scatter_nd_min(self.handle, indices, ops.convert_to_tensor(updates, self.dtype), name=name))

    def _write_object_proto(self, proto, options):
        """Writes additional information of the variable into the SavedObject proto.

    Subclasses of ResourceVariables could choose to override this method to
    customize extra information to provide when saving a SavedModel.

    Ideally, this should contain the logic in
    write_object_proto_for_resource_variable but `DistributedValue` is an
    outlier at the momemnt. Once `DistributedValue` becomes a proper
    ResourceVariable, we should remove the helper method below.

    Args:
      proto: `SavedObject` proto to update.
      options: A `SaveOption` instance that configures save behavior.
    """
        write_object_proto_for_resource_variable(self, proto, options)

    def _strided_slice_assign(self, begin, end, strides, value, name, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask):
        with _handle_graph(self.handle), self._assign_dependencies():
            return self._lazy_read(gen_array_ops.resource_strided_slice_assign(ref=self.handle, begin=begin, end=end, strides=strides, value=ops.convert_to_tensor(value, dtype=self.dtype), name=name, begin_mask=begin_mask, end_mask=end_mask, ellipsis_mask=ellipsis_mask, new_axis_mask=new_axis_mask, shrink_axis_mask=shrink_axis_mask))

    def __complex__(self):
        return complex(self.value().numpy())

    def __int__(self):
        return int(self.value().numpy())

    def __long__(self):
        return long(self.value().numpy())

    def __float__(self):
        return float(self.value().numpy())

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        del name
        if dtype is not None and (not dtype.is_compatible_with(self.dtype)):
            raise ValueError(f'Incompatible type conversion requested to type {dtype.name} for `tf.Variable of type {self.dtype.name}. (Variable: {self})')
        if as_ref:
            return self.read_value().op.inputs[0]
        else:
            return self.value()

    def __iadd__(self, unused_other):
        raise RuntimeError('`variable += value` with `tf.Variable`s is not supported. Use `variable.assign_add(value)` to modify the variable, or `out = variable + value` if you need to get a new output Tensor.')

    def __isub__(self, unused_other):
        raise RuntimeError('`variable -= value` with `tf.Variable`s is not supported. Use `variable.assign_sub(value)` to modify the variable, or `out = variable * value` if you need to get a new output Tensor.')

    def __imul__(self, unused_other):
        raise RuntimeError('`var *= value` with `tf.Variable`s is not supported. Use `var.assign(var * value)` to modify the variable, or `out = var * value` if you need to get a new output Tensor.')

    def __idiv__(self, unused_other):
        raise RuntimeError('`var /= value` with `tf.Variable`s is not supported. Use `var.assign(var / value)` to modify the variable, or `out = var / value` if you need to get a new output Tensor.')

    def __itruediv__(self, unused_other):
        raise RuntimeError('`var /= value` with `tf.Variable`s is not supported. Use `var.assign(var / value)` to modify the variable, or `out = var / value` if you need to get a new output Tensor.')

    def __irealdiv__(self, unused_other):
        raise RuntimeError('`var /= value` with `tf.Variable`s is not supported. Use `var.assign(var / value)` to modify the variable, or `out = var / value` if you need to get a new output Tensor.')

    def __ipow__(self, unused_other):
        raise RuntimeError('`var **= value` with `tf.Variable`s is not supported. Use `var.assign(var ** value)` to modify the variable, or `out = var ** value` if you need to get a new output Tensor.')