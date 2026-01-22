import functools
import traceback
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
class _EagerTemplateVariableStore:
    """Wrapper around EagerVariableStore to support nesting EagerTemplates."""

    def __init__(self, variable_scope_name):
        self._variable_scope_name = variable_scope_name
        default = variable_scope._get_default_variable_store()
        if default._store_eager_variables:
            self._eager_variable_store = variable_scope.EagerVariableStore(default)
        else:
            self._eager_variable_store = variable_scope.EagerVariableStore()
        self._used_once = False

    def set_variable_scope_name(self, variable_scope_name):
        self._variable_scope_name = variable_scope_name

    @tf_contextlib.contextmanager
    def as_default(self):
        try:
            if not self._used_once:
                default = variable_scope._get_default_variable_store()
                if default._store_eager_variables:
                    self._eager_variable_store._store = default
                self._used_once = True
            with self._eager_variable_store.as_default():
                yield
        finally:
            if self._variable_scope_name is None:
                raise RuntimeError('A variable scope must be set before an _EagerTemplateVariableStore object exits.')
            variable_scope.get_variable_scope_store().close_variable_subscopes(self._variable_scope_name)

    def _variables_in_scope(self, variable_list):
        if self._variable_scope_name is None:
            raise RuntimeError('A variable scope must be set before variables can be accessed.')
        return [v for v in variable_list if v.name.startswith(self._variable_scope_name + '/')]

    def variables(self):
        return self._variables_in_scope(self._eager_variable_store.variables())

    def trainable_variables(self):
        return self._variables_in_scope(self._eager_variable_store.trainable_variables())

    def non_trainable_variables(self):
        return self._variables_in_scope(self._eager_variable_store.non_trainable_variables())