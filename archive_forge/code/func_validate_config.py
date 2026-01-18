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
def validate_config(config):
    """Determines whether config appears to be a valid layer config."""
    return isinstance(config, dict) and _LAYER_UNDEFINED_CONFIG_KEY not in config