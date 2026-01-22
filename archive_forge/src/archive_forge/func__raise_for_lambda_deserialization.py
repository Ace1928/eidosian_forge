import inspect
import types
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import serialization_lib
from keras.src.utils import python_utils
from keras.src.utils import tree
@staticmethod
def _raise_for_lambda_deserialization(arg_name, safe_mode):
    if safe_mode:
        raise ValueError('The `{arg_name}` of this `Lambda` layer is a Python lambda. Deserializing it is unsafe. If you trust the source of the config artifact, you can override this error by passing `safe_mode=False` to `from_config()`, or calling `keras.config.enable_unsafe_deserialization().')