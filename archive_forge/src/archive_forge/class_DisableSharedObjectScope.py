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
class DisableSharedObjectScope(object):
    """A context manager for disabling handling of shared objects.

  Disables shared object handling for both saving and loading.

  Created primarily for use with `clone_model`, which does extra surgery that
  is incompatible with shared objects.
  """

    def __enter__(self):
        SHARED_OBJECT_DISABLED.disabled = True
        self._orig_loading_scope = _shared_object_loading_scope()
        self._orig_saving_scope = _shared_object_saving_scope()

    def __exit__(self, *args, **kwargs):
        SHARED_OBJECT_DISABLED.disabled = False
        SHARED_OBJECT_LOADING.scope = self._orig_loading_scope
        SHARED_OBJECT_SAVING.scope = self._orig_saving_scope