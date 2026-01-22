import collections
import copy
import os
from tensorflow.python.eager import def_function
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def _deserialize_nested_config(deserialize_fn, config):
    """Deserializes arbitrary Keras `config` using `deserialize_fn`."""

    def _is_single_object(obj):
        if isinstance(obj, dict) and 'class_name' in obj:
            return True
        if isinstance(obj, str):
            return True
        return False
    if config is None:
        return None
    if _is_single_object(config):
        return deserialize_fn(config)
    elif isinstance(config, dict):
        return {k: _deserialize_nested_config(deserialize_fn, v) for k, v in config.items()}
    elif isinstance(config, (tuple, list)):
        return [_deserialize_nested_config(deserialize_fn, obj) for obj in config]
    raise ValueError('Saved configuration not understood.')