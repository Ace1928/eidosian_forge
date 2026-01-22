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
def _raise_event(code, instruction, exc):
    """
    The way this should work is the following: when the user is using
    pydevd to do the launch and we're on a managed stack, we should consider
    unhandled only if it gets into a pydevd. If it's a thread, if it stops
    inside the threading and if it's an unmanaged thread (i.e.: QThread)
    then stop if it doesn't have a back frame.

    Note: unlike other events, this one is global and not per-code (so,
    it cannot be individually enabled/disabled for a given code object).
    """
    try:
        thread_info = _thread_local_info.thread_info
    except:
        thread_info = _get_thread_info(True, 1)
        if thread_info is None:
            return
    py_db: object = GlobalDebuggerHolder.global_dbg
    if py_db is None or py_db.pydb_disposed:
        return
    if not thread_info.trace or thread_info.thread._is_stopped:
        return
    func_code_info: FuncCodeInfo = _get_func_code_info(code, 1)
    if func_code_info.always_skip_code:
        return
    frame = _getframe(1)
    arg = (type(exc), exc, exc.__traceback__)
    should_stop, frame, _user_uncaught_exc_info = should_stop_on_exception(py_db, thread_info.additional_info, frame, thread_info.thread, arg, None)
    if should_stop:
        handle_exception(py_db, thread_info.thread, frame, arg, EXCEPTION_TYPE_HANDLED)
        return