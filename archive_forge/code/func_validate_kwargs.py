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
def validate_kwargs(kwargs, allowed_kwargs, error_message='Keyword argument not understood:'):
    """Checks that all keyword arguments are in the set of allowed keys."""
    for kwarg in kwargs:
        if kwarg not in allowed_kwargs:
            raise TypeError(error_message, kwarg)