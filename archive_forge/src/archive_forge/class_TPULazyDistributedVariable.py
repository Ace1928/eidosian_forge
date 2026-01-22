from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import tpu_replicated_variable
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
class TPULazyDistributedVariable(TPUDistributedVariable):
    """TPU Mirrored variable to be initialized lazily in a batch."""

    def _initialize_if_uninitialized(self):
        if getattr(self, '_is_lazily_initialized', False):
            return
        self._lazy_scope.initialize_all()
        self._is_lazily_initialized = True

    def assign_sub(self, value, use_locking=False, name=None, read_value=True):
        self._initialize_if_uninitialized()
        return super().assign_sub(value, use_locking, name, read_value)

    def assign_add(self, value, use_locking=False, name=None, read_value=True):
        self._initialize_if_uninitialized()
        return super().assign_add(value, use_locking, name, read_value)

    def assign(self, value, use_locking=False, name=None, read_value=True):
        self._initialize_if_uninitialized()
        return super().assign(value, use_locking, name, read_value)

    def read_value(self):
        self._initialize_if_uninitialized()
        return super().read_value()