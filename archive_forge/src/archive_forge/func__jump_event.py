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
def _jump_event(code, from_offset, to_offset):
    try:
        thread_info = _thread_local_info.thread_info
    except:
        thread_info = _get_thread_info(True, 1)
        if thread_info is None:
            return
    py_db: object = GlobalDebuggerHolder.global_dbg
    if py_db is None or py_db.pydb_disposed:
        return monitor.DISABLE
    if not thread_info.trace or thread_info.thread._is_stopped:
        return
    func_code_info: FuncCodeInfo = _get_func_code_info(code, 1)
    if func_code_info.always_skip_code or func_code_info.always_filtered_out:
        return monitor.DISABLE
    if to_offset > from_offset:
        return monitor.DISABLE
    from_line = func_code_info.get_line_of_offset(from_offset)
    to_line = func_code_info.get_line_of_offset(to_offset)
    if from_line != to_line:
        return monitor.DISABLE
    frame = _getframe(1)
    return _internal_line_event(func_code_info, frame, frame.f_lineno)