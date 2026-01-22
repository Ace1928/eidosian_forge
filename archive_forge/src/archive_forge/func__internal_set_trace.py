from _pydevd_bundle.pydevd_constants import get_frame, IS_CPYTHON, IS_64BIT_PROCESS, IS_WINDOWS, \
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import pydev_log, pydev_monkey
import os.path
import platform
import ctypes
from io import StringIO
import sys
import traceback
def _internal_set_trace(tracing_func):
    if TracingFunctionHolder._warn:
        frame = get_frame()
        if frame is not None and frame.f_back is not None:
            filename = os.path.splitext(frame.f_back.f_code.co_filename.lower())[0]
            if filename.endswith('threadpool') and 'gevent' in filename:
                if tracing_func is None:
                    pydev_log.debug('Disabled internal sys.settrace from gevent threadpool.')
                    return
            elif not filename.endswith(('threading', 'pydevd_tracing')):
                message = '\nPYDEV DEBUGGER WARNING:' + '\nsys.settrace() should not be used when the debugger is being used.' + '\nThis may cause the debugger to stop working correctly.' + '%s' % _get_stack_str(frame.f_back)
                if message not in TracingFunctionHolder._warnings_shown:
                    TracingFunctionHolder._warnings_shown[message] = 1
                    sys.stderr.write('%s\n' % (message,))
                    sys.stderr.flush()
    if TracingFunctionHolder._original_tracing:
        TracingFunctionHolder._original_tracing(tracing_func)