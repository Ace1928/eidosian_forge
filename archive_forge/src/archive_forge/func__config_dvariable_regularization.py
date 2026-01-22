import collections
import contextlib
import re
import threading
import tensorflow.compat.v2 as tf
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.dtensor import lazy_variable
from keras.src.dtensor import utils
from keras.src.engine import base_layer
from tensorflow.python.util.tf_export import keras_export
def _config_dvariable_regularization(layer, lazy_init_variable_to_tf_variable_map):
    """Update the weights regularizer for newly created `DVariable`.

    The weight regularization usually happens when `layer.add_weight()` is
    called, at which point the library will first create a `LazyInitVariable`,
    and then replace it with a `DVariable`. We will defer the creation of those
    losses, until the DVariable is created.

    See `layer._captured_weight_regularizer` for more details.

    Args:
      layer: the layer instance for DVariable regularization config.
      lazy_init_variable_to_tf_variable_map: the dict between LazyInitVariable
        ID and newly created DVariable.
    """
    for name, variable, regualarizer in layer._captured_weight_regularizer:
        if not _is_lazy_init_variable(variable):
            raise ValueError(f'Expect the regularization loss are created from LazyInitVariable, got {variable}')
        d_variable = lazy_init_variable_to_tf_variable_map[id(variable)]
        layer._handle_weight_regularization(name, d_variable, regualarizer)
    layer._captured_weight_regularizer = []