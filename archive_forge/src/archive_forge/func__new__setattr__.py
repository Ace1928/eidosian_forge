import copy
import sys
import textwrap
import traceback
import types
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import tf_decorator
def _new__setattr__(self, key, value):
    if key in ('_tf_should_use_helper', '_tf_should_use_wrapped_value'):
        return object.__setattr__(self, key, value)
    return setattr(object.__getattribute__(self, '_tf_should_use_wrapped_value'), key, value)