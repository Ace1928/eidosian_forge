import importlib
import inspect
import threading
import types
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.saving import object_registration
from keras.src.saving.legacy import serialization as legacy_serialization
from keras.src.saving.legacy.saved_model.utils import in_tf_saved_model_scope
from keras.src.utils import generic_utils
from tensorflow.python.util import tf_export
from tensorflow.python.util.tf_export import keras_export
def _get_class_or_fn_config(obj):
    """Return the object's config depending on its type."""
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    if hasattr(obj, 'get_config'):
        config = obj.get_config()
        if not isinstance(config, dict):
            raise TypeError(f'The `get_config()` method of {obj} should return a dict. It returned: {config}')
        return serialize_dict(config)
    elif hasattr(obj, '__name__'):
        return object_registration.get_registered_name(obj)
    else:
        raise TypeError(f'Cannot serialize object {obj} of type {type(obj)}. To be serializable, a class must implement the `get_config()` method.')