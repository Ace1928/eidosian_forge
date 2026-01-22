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
def _retrieve_class_or_fn(name, registered_name, module, obj_type, full_config, custom_objects=None):
    if obj_type == 'function':
        custom_obj = object_registration.get_registered_object(name, custom_objects=custom_objects)
    else:
        custom_obj = object_registration.get_registered_object(registered_name, custom_objects=custom_objects)
    if custom_obj is not None:
        return custom_obj
    if module:
        if module == 'keras' or module.startswith('keras.'):
            api_name = module + '.' + name
            if '__internal__.legacy' in api_name:
                api_name = 'compat.v1.' + api_name
            obj = tf_export.get_symbol_from_name(api_name)
            if obj is not None:
                return obj
        if obj_type == 'function' and module == 'builtins':
            for mod in BUILTIN_MODULES:
                obj = tf_export.get_symbol_from_name('keras.' + mod + '.' + name)
                if obj is not None:
                    return obj
            filtered_dict = {k: v for k, v in custom_objects.items() if k.endswith(full_config['config'])}
            if filtered_dict:
                return next(iter(filtered_dict.values()))
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError:
            raise TypeError(f"Could not deserialize {obj_type} '{name}' because its parent module {module} cannot be imported. Full object config: {full_config}")
        obj = vars(mod).get(name, None)
        if obj is None:
            if registered_name is not None:
                obj = vars(mod).get(registered_name, None)
            if name.count('.') == 1:
                outer_name, inner_name = name.split('.')
                outer_obj = vars(mod).get(outer_name, None)
                obj = getattr(outer_obj, inner_name, None) if outer_obj is not None else None
        if obj is not None:
            return obj
    raise TypeError(f"Could not locate {obj_type} '{name}'. Make sure custom classes are decorated with `@keras.saving.register_keras_serializable()`. Full object config: {full_config}")