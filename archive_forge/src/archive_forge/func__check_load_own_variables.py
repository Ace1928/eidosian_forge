import re
import string
import numpy as np
from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import ops
from keras.src import quantizers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
def _check_load_own_variables(self, store):
    all_vars = self._trainable_variables + self._non_trainable_variables
    if len(store.keys()) != len(all_vars):
        if len(all_vars) == 0 and (not self.built):
            raise ValueError(f"Layer '{self.name}' was never built and thus it doesn't have any variables. However the weights file lists {len(store.keys())} variables for this layer.\nIn most cases, this error indicates that either:\n\n1. The layer is owned by a parent layer that implements a `build()` method, but calling the parent's `build()` method did NOT create the state of the child layer '{self.name}'. A `build()` method must create ALL state for the layer, including the state of any children layers.\n\n2. You need to implement the `def build_from_config(self, config)` method on layer '{self.name}', to specify how to rebuild it during loading. In this case, you might also want to implement the method that generates the build config at saving time, `def get_build_config(self)`. The method `build_from_config()` is meant to create the state of the layer (i.e. its variables) upon deserialization.")
        raise ValueError(f"Layer '{self.name}' expected {len(all_vars)} variables, but received {len(store.keys())} variables during loading. Expected: {[v.name for v in all_vars]}")