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
def _shared_object_loading_scope():
    """Get the current shared object saving scope in a threadsafe manner."""
    return getattr(SHARED_OBJECT_LOADING, 'scope', NoopLoadingScope())