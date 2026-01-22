from _pydevd_bundle.pydevd_constants import get_frame, IS_CPYTHON, IS_64BIT_PROCESS, IS_WINDOWS, \
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import pydev_log, pydev_monkey
import os.path
import platform
import ctypes
from io import StringIO
import sys
import traceback
class TracingFunctionHolder:
    """This class exists just to keep some variables (so that we don't keep them in the global namespace).
    """
    _original_tracing = None
    _warn = True
    _traceback_limit = 1
    _warnings_shown = {}