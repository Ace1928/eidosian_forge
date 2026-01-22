import types
import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.name_scope import name_scope as base_name_scope
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.utils.naming import auto_name
class name_scope(base_name_scope):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._tf_name_scope = tf.name_scope(name)

    def __enter__(self):
        name_scope_stack = global_state.get_global_attribute('name_scope_stack', default=[], set_to_default=True)
        if self.deduplicate and name_scope_stack:
            parent_caller = name_scope_stack[-1].caller
            parent_name = name_scope_stack[-1].name
            if self.caller is not None and self.caller is parent_caller and (self.name == parent_name):
                return self
        name_scope_stack.append(self)
        self._pop_on_exit = True
        self._tf_name_scope.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        super().__exit__(*args, **kwargs)
        if self._pop_on_exit:
            self._tf_name_scope.__exit__(*args, **kwargs)