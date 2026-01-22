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
class SafeModeScope:
    """Scope to propagate safe mode flag to nested deserialization calls."""

    def __init__(self, safe_mode=True):
        self.safe_mode = safe_mode

    def __enter__(self):
        self.original_value = in_safe_mode()
        SAFE_MODE.safe_mode = self.safe_mode

    def __exit__(self, *args, **kwargs):
        SAFE_MODE.safe_mode = self.original_value