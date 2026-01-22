from __future__ import nested_scopes
import traceback
import warnings
from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import _pydev_saved_modules
import signal
import os
import ctypes
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from urllib.parse import quote  # @UnresolvedImport
import time
import inspect
import sys
from _pydevd_bundle.pydevd_constants import USE_CUSTOM_SYS_CURRENT_FRAMES, IS_PYPY, SUPPORT_GEVENT, \
def _extract_expression_list(log_message):
    expression = []
    expression_vars = []
    char_iter = iter(log_message)
    for c in char_iter:
        if c == '{':
            expression_var = _extract_variable_nested_braces(char_iter)
            if expression_var:
                expression.append('%s')
                expression_vars.append(expression_var)
        else:
            expression.append(c)
    expression = ''.join(expression)
    return (expression, expression_vars)