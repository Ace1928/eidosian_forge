import functools
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.module import module
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
class _EagerVariableStore(object):
    """TF2-compatible VariableStore that avoids collections & tracks regularizers.

  New variable names and new variables can be created; all stored
  variables are initialized with the initializer passed to __init__.

  All variables get created in `tf.init_scope.` to avoid a bad
  interaction between `tf.function` `FuncGraph` internals, Keras
  Functional Models, and TPUStrategy variable initialization.

  Attributes:
    vars: a dictionary with string names (same as passed in GetVar) as keys and
      the corresponding TensorFlow Variables as values.
  """
    __slots__ = ['_vars', '_regularizers', '_store_eager_variables']

    def __init__(self):
        """Create a variable store."""
        self._vars = {}
        self._regularizers = {}
        self._store_eager_variables = True

    def get_variable(self, name, shape=None, dtype=dtypes.float32, initializer=None, regularizer=None, reuse=None, trainable=None, collections=None, caching_device=None, partitioner=None, validate_shape=True, use_resource=None, custom_getter=None, constraint=None, synchronization=vs.VariableSynchronization.AUTO, aggregation=vs.VariableAggregation.NONE):
        """Gets an existing variable with these parameters or create a new one.

    If a variable with the given name is already stored, we return the stored
    variable. Otherwise, we create a new one.

    Set `reuse` to `True` when you only want to reuse existing Variables.
    Set `reuse` to `False` when you only want to create new Variables.
    Set `reuse` to None (the default) or tf.compat.v1.AUTO_REUSE when you want
    variables to be created if they don't exist or returned if they do.

    If initializer is `None` (the default), the default initializer passed in
    the constructor is used. If that one is `None` too, we use a new
    `glorot_uniform_initializer`. If initializer is a Tensor, we use
    it as a value and derive the shape from the initializer.

    If a partitioner is provided, a `PartitionedVariable` is returned.
    Accessing this object as a `Tensor` returns the shards concatenated along
    the partition axis.

    Some useful partitioners are available.  See, e.g.,
    `variable_axis_size_partitioner` and `min_max_variable_partitioner`.

    Args:
      name: The name of the new or existing variable.
      shape: Shape of the new or existing variable.
      dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).
      initializer: Initializer for the variable.
      regularizer: A (Tensor -> Tensor or None) function; the result of applying
        it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
      reuse: a Boolean, None, or tf.AUTO_REUSE. Controls reuse or creation of
        variables. When eager execution is enabled  this argument is always
        forced to be False.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`). `trainable`
        defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys to add the `Variable` to.
        Defaults to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the `Variable` reside, to
        deduplicate copying through `Switch` and other conditional statements.
      partitioner: Optional callable that accepts a fully defined `TensorShape`
        and dtype of the `Variable` to be created, and returns a list of
        partitions for each axis (currently only one axis can be partitioned).
      validate_shape: If False, allows the variable to be initialized with a
        value of unknown shape. If True, the default, the shape of initial_value
        must be known.
      use_resource: If False, creates a regular Variable. If True, creates
        instead an experimental ResourceVariable which has well-defined
        semantics. Defaults to False (will later change to True). When eager
        execution is enabled this argument is always forced to be true.
      custom_getter: Callable that takes as a first argument the true getter,
        and allows overwriting the internal get_variable method. The signature
        of `custom_getter` should match that of this method,
        but the most future-proof version will allow for changes: `def
          custom_getter(getter, *args, **kwargs)`.  Direct access to
        all `get_variable` parameters is also allowed: `def
          custom_getter(getter, name, *args, **kwargs)`.  A simple identity
        custom getter that simply creates variables with modified names is:
          ```python
        def custom_getter(getter, name, *args, **kwargs): return getter(name +
          '_suffix', *args, **kwargs) ```
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

    Returns:
      The created or existing `Variable` (or `PartitionedVariable`, if a
      partitioner was used).

    Raises:
      ValueError: when creating a new variable and shape is not declared,
        when reusing a variable and specifying a conflicting shape,
        or when violating reuse during variable creation.
      RuntimeError: when eager execution is enabled and not called from an
        EagerVariableStore.
    """
        if custom_getter is not None and (not callable(custom_getter)):
            raise ValueError('Passed a custom_getter which is not callable: %s' % custom_getter)
        with ops.init_scope():
            if context.executing_eagerly():
                use_resource = True
        if context.executing_eagerly():
            reuse = vs.AUTO_REUSE
        try:
            dtype = dtype.base_dtype
        except AttributeError:
            pass

        def _true_getter(name, shape=None, dtype=dtypes.float32, initializer=None, regularizer=None, reuse=None, trainable=None, collections=None, caching_device=None, partitioner=None, validate_shape=True, use_resource=None, constraint=None, synchronization=vs.VariableSynchronization.AUTO, aggregation=vs.VariableAggregation.NONE):
            if partitioner is not None:
                raise ValueError('`partitioner` arg for `get_variable` is unsupported in TF2.File a bug if you need help. You passed %s' % partitioner)
            if '%s/part_0' % name in self._vars:
                raise ValueError('No partitioner was provided, but a partitioned version of the variable was found: %s/part_0. Perhaps a variable of the same name was already created with partitioning?' % name)
            return self._get_single_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, reuse=reuse, trainable=trainable, caching_device=caching_device, validate_shape=validate_shape, constraint=constraint, synchronization=synchronization, aggregation=aggregation)
        synchronization, aggregation, trainable = validate_synchronization_aggregation_trainable(synchronization, aggregation, trainable, name)
        if custom_getter is not None:
            custom_getter_kwargs = {'getter': _true_getter, 'name': name, 'shape': shape, 'dtype': dtype, 'initializer': initializer, 'regularizer': regularizer, 'reuse': reuse, 'trainable': trainable, 'collections': collections, 'caching_device': caching_device, 'partitioner': partitioner, 'validate_shape': validate_shape, 'use_resource': use_resource, 'synchronization': synchronization, 'aggregation': aggregation}
            if 'constraint' in fn_args(custom_getter) or _has_kwargs(custom_getter):
                custom_getter_kwargs['constraint'] = constraint
            return custom_getter(**custom_getter_kwargs)
        else:
            return _true_getter(name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, reuse=reuse, trainable=trainable, collections=collections, caching_device=caching_device, partitioner=partitioner, validate_shape=validate_shape, use_resource=use_resource, constraint=constraint, synchronization=synchronization, aggregation=aggregation)

    def _get_single_variable(self, name, shape=None, dtype=dtypes.float32, initializer=None, regularizer=None, partition_info=None, reuse=None, trainable=None, caching_device=None, validate_shape=True, constraint=None, synchronization=vs.VariableSynchronization.AUTO, aggregation=vs.VariableAggregation.NONE):
        """Get or create a single Variable (e.g.

    a shard or entire variable).

    See the documentation of get_variable above (ignore partitioning components)
    for details.

    Args:
      name: see get_variable.
      shape: see get_variable.
      dtype: see get_variable.
      initializer: see get_variable.
      regularizer: see get_variable.
      partition_info: _PartitionInfo object.
      reuse: see get_variable.
      trainable: see get_variable.
      caching_device: see get_variable.
      validate_shape: see get_variable.
      constraint: see get_variable.
      synchronization: see get_variable.
      aggregation: see get_variable.

    Returns:
      A Variable.  See documentation of get_variable above.

    Raises:
      ValueError: See documentation of get_variable above.
    """
        initializing_from_value = False
        if initializer is not None and (not callable(initializer)):
            initializing_from_value = True
        if shape is not None and initializing_from_value:
            raise ValueError('If initializer is a constant, do not specify shape.')
        dtype = dtypes.as_dtype(dtype)
        shape = as_shape(shape)
        if name in self._vars:
            if reuse is False:
                err_msg = 'Variable %s already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?' % name
                raise ValueError(err_msg)
            found_var = self._vars[name]
            if not shape.is_compatible_with(found_var.get_shape()):
                raise ValueError('Trying to share variable %s, but specified shape %s and found shape %s.' % (name, shape, found_var.get_shape()))
            if not dtype.is_compatible_with(found_var.dtype):
                dtype_str = dtype.name
                found_type_str = found_var.dtype.name
                raise ValueError('Trying to share variable %s, but specified dtype %s and found dtype %s.' % (name, dtype_str, found_type_str))
            return found_var
        if reuse is True:
            raise ValueError('Variable %s does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=tf.AUTO_REUSE in VarScope?' % name)
        if initializer is None:
            initializer, initializing_from_value = self._get_default_initializer(name=name, shape=shape, dtype=dtype)
        with ops.init_scope():
            if initializing_from_value:
                init_val = initializer
                variable_dtype = None
            else:
                if tf_inspect.isclass(initializer):
                    initializer = initializer()
                if shape.is_fully_defined():
                    if 'partition_info' in tf_inspect.getargspec(initializer).args:
                        init_val = functools.partial(initializer, shape.as_list(), dtype=dtype, partition_info=partition_info)
                    else:
                        init_val = functools.partial(initializer, shape.as_list(), dtype=dtype)
                    variable_dtype = dtype.base_dtype
                else:
                    init_val = initializer
                    variable_dtype = None
        with ops.init_scope():
            v = variables.Variable(initial_value=init_val, name=name, trainable=trainable, caching_device=caching_device, dtype=variable_dtype, validate_shape=validate_shape, constraint=constraint, synchronization=synchronization, aggregation=aggregation)
        self._vars[name] = v
        logging.vlog(1, 'Created variable %s with shape %s and init %s', v.name, format(shape), initializer)
        if regularizer:
            self.add_regularizer(v, regularizer)
        return v

    def add_regularizer(self, var, regularizer):
        self._regularizers[var.name] = functools.partial(regularizer, var)

    def _get_default_initializer(self, name, shape=None, dtype=dtypes.float32):
        """Provide a default initializer and a corresponding value.

    Args:
      name: see get_variable.
      shape: see get_variable.
      dtype: see get_variable.

    Returns:
      initializer and initializing_from_value. See get_variable above.

    Raises:
      ValueError: When giving unsupported dtype.
    """
        del shape
        if dtype.is_floating:
            initializer = init_ops.glorot_uniform_initializer()
            initializing_from_value = False
        elif dtype.is_integer or dtype.is_unsigned or dtype.is_bool or (dtype == dtypes.string):
            initializer = init_ops.zeros_initializer()
            initializing_from_value = False
        else:
            raise ValueError('An initializer for variable %s of %s is required' % (name, dtype.base_dtype))
        return (initializer, initializing_from_value)