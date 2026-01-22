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
def _compute_repr_slow(self, diff, attrs_tab_separated, attr_name, attr_type):
    try:
        attr_type = attr_type.__name__
    except:
        pass
    if attrs_tab_separated:
        return 'pydevd warning: Computing repr of %s.%s (%s) was slow (took %.2fs).\nCustomize report timeout by setting the `PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT` environment variable to a higher timeout (default is: %ss)\n' % (attrs_tab_separated.replace('\t', '.'), attr_name, attr_type, diff, PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT)
    else:
        return 'pydevd warning: Computing repr of %s (%s) was slow (took %.2fs)\nCustomize report timeout by setting the `PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT` environment variable to a higher timeout (default is: %ss)\n' % (attr_name, attr_type, diff, PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT)