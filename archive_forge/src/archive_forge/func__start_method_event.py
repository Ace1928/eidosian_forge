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
def _start_method_event(code, instruction_offset):
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
    frame = _getframe(1)
    func_code_info = _get_func_code_info(code, frame)
    if func_code_info.always_skip_code:
        return monitor.DISABLE
    keep_enabled: bool = _enable_code_tracing(py_db, thread_info.additional_info, func_code_info, code, frame, True)
    if func_code_info.function_breakpoint_found:
        bp = func_code_info.function_breakpoint
        stop = True
        new_frame = frame
        stop_reason = CMD_SET_FUNCTION_BREAK
        stop_on_plugin_breakpoint = False
        _stop_on_breakpoint(py_db, thread_info, stop_reason, bp, frame, new_frame, stop, stop_on_plugin_breakpoint, 'python-function')
        return
    if py_db.plugin:
        plugin_manager = py_db.plugin
        info = thread_info.additional_info
        if func_code_info.plugin_call_breakpoint_found:
            result = plugin_manager.get_breakpoint(py_db, frame, 'call', info)
            if result:
                stop_reason = CMD_SET_BREAK
                stop = False
                stop_on_plugin_breakpoint = True
                bp, new_frame, bp_type = result
                _stop_on_breakpoint(py_db, thread_info, stop_reason, bp, frame, new_frame, stop, stop_on_plugin_breakpoint, bp_type)
                return
            keep_enabled = True
        step_cmd = info.pydev_step_cmd
        if step_cmd != -1 and func_code_info.plugin_call_stepping and (info.suspend_type != PYTHON_SUSPEND):
            _plugin_stepping(py_db, step_cmd, 'call', frame, thread_info)
            return
    if keep_enabled or any_thread_stepping():
        return None
    return monitor.DISABLE