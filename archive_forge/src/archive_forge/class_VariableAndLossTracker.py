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
class VariableAndLossTracker(module.Module):
    """Module that has a scope to capture vars/losses made by `get_variable`."""

    def __init__(self):
        self._var_store = _EagerVariableStore()
        self._variables = {}

    def _variable_creator(self, next_creator, **kwargs):
        var = next_creator(**kwargs)
        self._variables[var.name] = var
        return var

    @tf_contextlib.contextmanager
    def scope(self):
        with vs.variable_creator_scope(self._variable_creator), vs.with_variable_store(self._var_store):
            yield

    def get_regularization_losses(self):
        losses = {}
        for var_name, regularizer in self._var_store._regularizers.items():
            losses[var_name] = regularizer()
        return losses