from collections import namedtuple
import dis
import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from types import CodeType, FrameType
from typing import Dict, Optional, Tuple, Any
from os.path import basename, splitext
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_dont_trace
from _pydevd_bundle.pydevd_constants import (GlobalDebuggerHolder, ForkSafeLock,
from pydevd_file_utils import (NORM_PATHS_AND_BASE_CONTAINER,
from _pydevd_bundle.pydevd_trace_dispatch import should_stop_on_exception, handle_exception
from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_HANDLED
from _pydevd_bundle.pydevd_trace_dispatch import is_unhandled_exception
from _pydevd_bundle.pydevd_breakpoints import stop_on_unhandled_exception
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info, any_thread_stepping, PyDBAdditionalThreadInfo
def _stop_on_return(py_db, thread_info, info, step_cmd, frame, retval):
    back = frame.f_back
    if back is not None:
        back_absolute_filename, _, base = get_abs_path_real_path_and_base_from_frame(back)
        if (base, back.f_code.co_name) in (DEBUG_START, DEBUG_START_PY3K):
            back = None
        elif base == TRACE_PROPERTY:
            return
        elif pydevd_dont_trace.should_trace_hook is not None:
            if not pydevd_dont_trace.should_trace_hook(back.f_code, back_absolute_filename):
                py_db.set_trace_for_frame_and_parents(thread_info.thread_ident, back)
                return
    if back is not None:
        py_db.set_suspend(thread_info.thread, step_cmd, original_step_cmd=info.pydev_original_step_cmd)
        _do_wait_suspend(py_db, thread_info, back, 'return', retval)
    else:
        info.pydev_step_stop = None
        info.pydev_original_step_cmd = -1
        info.pydev_step_cmd = -1
        info.pydev_state = STATE_RUN
        info.update_stepping_info()