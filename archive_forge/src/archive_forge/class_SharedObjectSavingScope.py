import binascii
import codecs
import importlib
import marshal
import os
import re
import sys
import threading
import time
import types as python_types
import warnings
import weakref
import numpy as np
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
class SharedObjectSavingScope(object):
    """Keeps track of shared object configs when serializing."""

    def __enter__(self):
        if _shared_object_disabled():
            return None
        global SHARED_OBJECT_SAVING
        if _shared_object_saving_scope() is not None:
            self._passthrough = True
            return _shared_object_saving_scope()
        else:
            self._passthrough = False
        SHARED_OBJECT_SAVING.scope = self
        self._shared_objects_config = weakref.WeakKeyDictionary()
        self._next_id = 0
        return self

    def get_config(self, obj):
        """Gets a `SharedObjectConfig` if one has already been seen for `obj`.

    Args:
      obj: The object for which to retrieve the `SharedObjectConfig`.

    Returns:
      The SharedObjectConfig for a given object, if already seen. Else,
        `None`.
    """
        try:
            shared_object_config = self._shared_objects_config[obj]
        except (TypeError, KeyError):
            return None
        shared_object_config.increment_ref_count()
        return shared_object_config

    def create_config(self, base_config, obj):
        """Create a new SharedObjectConfig for a given object."""
        shared_object_config = SharedObjectConfig(base_config, self._next_id)
        self._next_id += 1
        try:
            self._shared_objects_config[obj] = shared_object_config
        except TypeError:
            pass
        return shared_object_config

    def __exit__(self, *args, **kwargs):
        if not getattr(self, '_passthrough', False):
            global SHARED_OBJECT_SAVING
            SHARED_OBJECT_SAVING.scope = None