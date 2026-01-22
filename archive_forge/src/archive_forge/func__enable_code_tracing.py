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
def _enable_code_tracing(py_db, additional_info, func_code_info: FuncCodeInfo, code, frame, warn_on_filtered_out) -> bool:
    """
    :return: Whether code tracing was added in this function to the given code.
    """
    step_cmd = additional_info.pydev_step_cmd
    is_stepping = step_cmd != -1
    code_tracing_added = False
    if func_code_info.always_filtered_out:
        if warn_on_filtered_out and is_stepping and (additional_info.pydev_original_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE)) and (not _global_notify_skipped_step_in):
            _notify_skipped_step_in_because_of_filters(py_db, frame)
        if is_stepping:
            _enable_step_tracing(py_db, code, step_cmd, additional_info, frame)
            code_tracing_added = True
        return code_tracing_added
    if func_code_info.breakpoint_found or func_code_info.plugin_line_breakpoint_found:
        _enable_line_tracing(code)
        code_tracing_added = True
    if is_stepping:
        _enable_step_tracing(py_db, code, step_cmd, additional_info, frame)
        code_tracing_added = True
    return code_tracing_added