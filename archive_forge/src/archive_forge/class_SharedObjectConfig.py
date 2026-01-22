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
class SharedObjectConfig(dict):
    """A configuration container that keeps track of references.

  `SharedObjectConfig` will automatically attach a shared object ID to any
  configs which are referenced more than once, allowing for proper shared
  object reconstruction at load time.

  In most cases, it would be more proper to subclass something like
  `collections.UserDict` or `collections.Mapping` rather than `dict` directly.
  Unfortunately, python's json encoder does not support `Mapping`s. This is
  important functionality to retain, since we are dealing with serialization.

  We should be safe to subclass `dict` here, since we aren't actually
  overriding any core methods, only augmenting with a new one for reference
  counting.
  """

    def __init__(self, base_config, object_id, **kwargs):
        self.ref_count = 1
        self.object_id = object_id
        super(SharedObjectConfig, self).__init__(base_config, **kwargs)

    def increment_ref_count(self):
        if self.ref_count == 1:
            self[SHARED_OBJECT_KEY] = self.object_id
        self.ref_count += 1