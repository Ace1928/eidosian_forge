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
def _deferred_initialize(self):
    if self._value is not None:
        raise ValueError(f'Variable {self.path} is already initialized.')
    if in_stateless_scope():
        raise ValueError('You are attempting to initialize a variable while in a stateless scope. This is disallowed. Make sure that all variables are initialized before you start using your layer/model objects.')
    with tf.init_scope():
        value = self._initializer(self._shape, dtype=self._dtype)
        self._initialize(value)