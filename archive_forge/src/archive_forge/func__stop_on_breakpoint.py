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
def _stop_on_breakpoint(py_db, thread_info: ThreadInfo, stop_reason: int, bp, frame, new_frame, stop: bool, stop_on_plugin_breakpoint: bool, bp_type: str):
    """
    :param bp: the breakpoint hit (additional conditions will be checked now).
    :param frame: the actual frame
    :param new_frame: either the actual frame or the frame provided by the plugins.
    :param stop: whether we should do a regular line breakpoint.
    :param stop_on_plugin_breakpoint: whether we should stop in a plugin breakpoint.

    :return:
        True if the breakpoint was suspended inside this function and False otherwise.
        Note that even if False is returned, it's still possible
    """
    additional_info = thread_info.additional_info
    if bp.expression is not None:
        py_db.handle_breakpoint_expression(bp, additional_info, new_frame)
    if stop or stop_on_plugin_breakpoint:
        if bp.has_condition:
            eval_result = py_db.handle_breakpoint_condition(additional_info, bp, new_frame)
            if not eval_result:
                stop = False
                stop_on_plugin_breakpoint = False
    if (stop or stop_on_plugin_breakpoint) and bp.is_logpoint:
        stop = False
        stop_on_plugin_breakpoint = False
        if additional_info.pydev_message is not None and len(additional_info.pydev_message) > 0:
            cmd = py_db.cmd_factory.make_io_message(additional_info.pydev_message + os.linesep, '1')
            py_db.writer.add_command(cmd)
    if stop:
        py_db.set_suspend(thread_info.thread, stop_reason, suspend_other_threads=bp and bp.suspend_policy == 'ALL')
        _do_wait_suspend(py_db, thread_info, frame, 'line', None)
        return True
    elif stop_on_plugin_breakpoint:
        stop_at_frame = py_db.plugin.suspend(py_db, thread_info.thread, frame, bp_type)
        if stop_at_frame and thread_info.additional_info.pydev_state == STATE_SUSPEND:
            _do_wait_suspend(py_db, thread_info, stop_at_frame, 'line', None)
        return
    return False