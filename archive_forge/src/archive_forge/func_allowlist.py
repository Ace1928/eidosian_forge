import contextlib
import imp
import inspect
import io
import sys
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import api
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
def allowlist(f):
    """Helper that marks a callable as whtelitisted."""
    if 'allowlisted_module_for_testing' not in sys.modules:
        allowlisted_mod = imp.new_module('allowlisted_module_for_testing')
        sys.modules['allowlisted_module_for_testing'] = allowlisted_mod
        config.CONVERSION_RULES = (config.DoNotConvert('allowlisted_module_for_testing'),) + config.CONVERSION_RULES
    f.__module__ = 'allowlisted_module_for_testing'