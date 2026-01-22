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
class ScopeRequest(object):
    __slots__ = ['variable_reference', 'scope']

    def __init__(self, variable_reference, scope):
        assert scope in ('globals', 'locals')
        self.variable_reference = variable_reference
        self.scope = scope

    def __eq__(self, o):
        if isinstance(o, ScopeRequest):
            return self.variable_reference == o.variable_reference and self.scope == o.scope
        return False

    def __ne__(self, o):
        return not self == o

    def __hash__(self):
        return hash((self.variable_reference, self.scope))