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
def _map_functional_model_variable(model, layout_map):
    """Map/Replace LazyInitVariable for functional/sequential model."""
    lazy_init_variable_to_tf_variable_map = {}
    for layer in model.layers:
        layer_name = layer.name
        for path, variable in layer._flatten(predicate=_is_lazy_init_variable, with_path=True):
            if [a for a in _KERAS_ATTRIBUTES_TO_SKIP if a in path]:
                continue
            object_path = '.'.join([str(item) for item in path])
            object_path = layer_name + '.' + object_path
            new_variable = _create_dvariable(layout_map, object_path, variable)
            _set_object_by_path(layer, path, new_variable)
            lazy_init_variable_to_tf_variable_map[id(variable)] = new_variable
        _config_dvariable_regularization(layer, lazy_init_variable_to_tf_variable_map)
        for path, variable in layer._flatten(predicate=_is_lazy_init_variable, with_path=True):
            tf_variable = lazy_init_variable_to_tf_variable_map[id(variable)]
            _set_object_by_path(layer, path, tf_variable)
    _init_state_variable_for_rng(model, layout_map)
    _update_trackable_reference(model, lazy_init_variable_to_tf_variable_map)
    return model