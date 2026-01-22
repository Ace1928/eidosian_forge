import copy
import enum
import functools
import sys
import threading
import traceback
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class _pure_variable_scope:
    """A context for the variable_scope, see `variable_scope` for docs."""

    def __init__(self, name_or_scope, reuse=None, initializer=None, regularizer=None, caching_device=None, partitioner=None, custom_getter=None, old_name_scope=None, dtype=dtypes.float32, use_resource=None, constraint=None):
        """Creates a context for the variable_scope, see `variable_scope` for docs.

    Note: this does not create a name scope.

    Args:
      name_or_scope: `string` or `VariableScope`: the scope to open.
      reuse: `True` or None, or tf.compat.v1.AUTO_REUSE; if `None`, we inherit
        the parent scope's reuse flag.
      initializer: default initializer for variables within this scope.
      regularizer: default regularizer for variables within this scope.
      caching_device: default caching device for variables within this scope.
      partitioner: default partitioner for variables within this scope.
      custom_getter: default custom getter for variables within this scope.
      old_name_scope: the original name scope when re-entering a variable scope.
      dtype: type of the variables within this scope (defaults to `DT_FLOAT`).
      use_resource: If False, variables in this scope will be regular Variables.
        If True, experimental ResourceVariables will be creates instead, with
        well-defined semantics. Defaults to False (will later change to True).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
    """
        self._name_or_scope = name_or_scope
        self._reuse = reuse
        self._initializer = initializer
        self._regularizer = regularizer
        self._caching_device = caching_device
        self._partitioner = partitioner
        self._custom_getter = custom_getter
        self._old_name_scope = old_name_scope
        self._dtype = dtype
        self._use_resource = use_resource
        self._constraint = constraint
        self._var_store = _get_default_variable_store()
        self._var_scope_store = get_variable_scope_store()
        self._last_variable_scope_object = None
        if isinstance(self._name_or_scope, VariableScope):
            self._new_name = self._name_or_scope.name
            name_scope = self._name_or_scope._name_scope
            variable_scope_object = VariableScope(self._name_or_scope.reuse if not self._reuse else self._reuse, name=self._new_name, initializer=self._name_or_scope.initializer, regularizer=self._name_or_scope.regularizer, caching_device=self._name_or_scope.caching_device, partitioner=self._name_or_scope.partitioner, dtype=self._name_or_scope.dtype, custom_getter=self._name_or_scope.custom_getter, name_scope=name_scope, use_resource=self._name_or_scope.use_resource, constraint=self._constraint)
            if self._initializer is not None:
                variable_scope_object.set_initializer(self._initializer)
            if self._regularizer is not None:
                variable_scope_object.set_regularizer(self._regularizer)
            if self._caching_device is not None:
                variable_scope_object.set_caching_device(self._caching_device)
            if self._partitioner is not None:
                variable_scope_object.set_partitioner(self._partitioner)
            if self._custom_getter is not None:
                variable_scope_object.set_custom_getter(_maybe_wrap_custom_getter(self._custom_getter, self._name_or_scope.custom_getter))
            if self._dtype is not None:
                variable_scope_object.set_dtype(self._dtype)
            if self._use_resource is not None:
                variable_scope_object.set_use_resource(self._use_resource)
            self._cached_variable_scope_object = variable_scope_object

    def __enter__(self):
        """Begins the scope block.

    Returns:
      A VariableScope.
    Raises:
      ValueError: when trying to reuse within a create scope, or create within
        a reuse scope, or if reuse is not `None` or `True`.
      TypeError: when the types of some arguments are not appropriate.
    """
        self._old = self._var_scope_store.current_scope
        if isinstance(self._name_or_scope, VariableScope):
            self._var_scope_store.open_variable_scope(self._new_name)
            self._old_subscopes = copy.copy(self._var_scope_store.variable_scopes_count)
            variable_scope_object = self._cached_variable_scope_object
        else:
            self._new_name = self._old.name + '/' + self._name_or_scope if self._old.name else self._name_or_scope
            self._reuse = self._reuse or self._old.reuse
            if self._old_name_scope is None:
                name_scope = self._name_or_scope
            else:
                name_scope = self._old_name_scope
            variable_scope_object = VariableScope(self._reuse, name=self._new_name, initializer=self._old.initializer, regularizer=self._old.regularizer, caching_device=self._old.caching_device, partitioner=self._old.partitioner, dtype=self._old.dtype, use_resource=self._old.use_resource, custom_getter=self._old.custom_getter, name_scope=name_scope, constraint=self._constraint)
            if self._initializer is not None:
                variable_scope_object.set_initializer(self._initializer)
            if self._regularizer is not None:
                variable_scope_object.set_regularizer(self._regularizer)
            if self._caching_device is not None:
                variable_scope_object.set_caching_device(self._caching_device)
            if self._partitioner is not None:
                variable_scope_object.set_partitioner(self._partitioner)
            if self._custom_getter is not None:
                variable_scope_object.set_custom_getter(_maybe_wrap_custom_getter(self._custom_getter, self._old.custom_getter))
            if self._dtype is not None:
                variable_scope_object.set_dtype(self._dtype)
            if self._use_resource is not None:
                variable_scope_object.set_use_resource(self._use_resource)
            self._var_scope_store.open_variable_scope(self._new_name)
        self._var_scope_store.current_scope = variable_scope_object
        self._last_variable_scope_object = variable_scope_object
        return variable_scope_object

    def __exit__(self, type_arg, value_arg, traceback_arg):
        if self._var_scope_store.current_scope is not self._last_variable_scope_object:
            raise RuntimeError('Improper nesting of variable_scope.')
        if isinstance(self._name_or_scope, VariableScope):
            self._var_scope_store.variable_scopes_count = self._old_subscopes
        else:
            self._var_scope_store.close_variable_subscopes(self._new_name)
        self._var_scope_store.current_scope = self._old