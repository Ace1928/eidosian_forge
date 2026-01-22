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
def _internal_line_event(func_code_info, frame, line):
    py_db: object = GlobalDebuggerHolder.global_dbg
    thread_info = _thread_local_info.thread_info
    info = thread_info.additional_info
    step_cmd = info.pydev_step_cmd
    if func_code_info.breakpoint_found:
        bp = None
        stop = False
        stop_on_plugin_breakpoint = False
        stop_info = {}
        stop_reason = CMD_SET_BREAK
        bp_type = None
        bp = func_code_info.bp_line_to_breakpoint.get(line)
        if bp is not None:
            new_frame = frame
            stop = True
        if bp:
            if _stop_on_breakpoint(py_db, thread_info, stop_reason, bp, frame, new_frame, stop, stop_on_plugin_breakpoint, 'python-line'):
                return
    if func_code_info.plugin_line_breakpoint_found:
        result = py_db.plugin.get_breakpoint(py_db, frame, 'line', info)
        if result:
            stop_reason = CMD_SET_BREAK
            stop = False
            stop_on_plugin_breakpoint = True
            bp, new_frame, bp_type = result
            _stop_on_breakpoint(py_db, thread_info, stop_reason, bp, frame, new_frame, stop, stop_on_plugin_breakpoint, bp_type)
            return
    if info.pydev_state == STATE_SUSPEND:
        _do_wait_suspend(py_db, thread_info, frame, 'line', None)
        return
    stop_frame = info.pydev_step_stop
    if step_cmd == -1:
        if func_code_info.breakpoint_found or func_code_info.plugin_line_breakpoint_found or any_thread_stepping():
            return None
        return monitor.DISABLE
    if info.suspend_type != PYTHON_SUSPEND:
        if func_code_info.plugin_line_stepping:
            _plugin_stepping(py_db, step_cmd, 'line', frame, thread_info)
        return
    if step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE, CMD_STEP_INTO_COROUTINE):
        force_check_project_scope = step_cmd == CMD_STEP_INTO_MY_CODE
        if not info.pydev_use_scoped_step_frame:
            if func_code_info.always_filtered_out or (force_check_project_scope and func_code_info.filtered_out_force_checked):
                return
            py_db.set_suspend(thread_info.thread, step_cmd, original_step_cmd=info.pydev_original_step_cmd)
            _do_wait_suspend(py_db, thread_info, frame, 'line', None)
            return
        else:
            if func_code_info.always_filtered_out or (force_check_project_scope and func_code_info.filtered_out_force_checked):
                return
            stop = False
            filename = frame.f_code.co_filename
            if filename.endswith('.pyc'):
                filename = filename[:-1]
            if not filename.endswith(PYDEVD_IPYTHON_CONTEXT[0]):
                f = frame.f_back
                while f is not None:
                    if f.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[1]:
                        f2 = f.f_back
                        if f2 is not None and f2.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[2]:
                            pydev_log.debug('Stop inside ipython call')
                            py_db.set_suspend(thread_info.thread, step_cmd, original_step_cmd=info.pydev_original_step_cmd)
                            thread_info.additional_info.trace_suspend_type = 'sys_monitor'
                            _do_wait_suspend(py_db, thread_info, frame, 'line', None)
                            break
                    f = f.f_back
                del f
        return
    elif step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE):
        if _is_same_frame(info, stop_frame, frame):
            py_db.set_suspend(thread_info.thread, step_cmd, original_step_cmd=info.pydev_original_step_cmd)
            _do_wait_suspend(py_db, thread_info, frame, 'line', None)
            return
    elif step_cmd == CMD_SMART_STEP_INTO:
        stop = False
        back = frame.f_back
        if _is_same_frame(info, stop_frame, back):
            if info.pydev_smart_child_offset != -1:
                stop = False
            else:
                pydev_smart_parent_offset = info.pydev_smart_parent_offset
                pydev_smart_step_into_variants = info.pydev_smart_step_into_variants
                if pydev_smart_parent_offset >= 0 and pydev_smart_step_into_variants:
                    stop = get_smart_step_into_variant_from_frame_offset(back.f_lasti, pydev_smart_step_into_variants) is get_smart_step_into_variant_from_frame_offset(pydev_smart_parent_offset, pydev_smart_step_into_variants)
                else:
                    curr_func_name = frame.f_code.co_name
                    if curr_func_name in ('?', '<module>') or curr_func_name is None:
                        curr_func_name = ''
                    if curr_func_name == info.pydev_func_name and stop_frame.f_lineno == info.pydev_next_line:
                        stop = True
            if not stop:
                return
        elif back is not None and _is_same_frame(info, stop_frame, back.f_back):
            pydev_smart_parent_offset = info.pydev_smart_parent_offset
            pydev_smart_child_offset = info.pydev_smart_child_offset
            stop = False
            if pydev_smart_child_offset >= 0 and pydev_smart_child_offset >= 0:
                pydev_smart_step_into_variants = info.pydev_smart_step_into_variants
                if pydev_smart_parent_offset >= 0 and pydev_smart_step_into_variants:
                    smart_step_into_variant = get_smart_step_into_variant_from_frame_offset(pydev_smart_parent_offset, pydev_smart_step_into_variants)
                    children_variants = smart_step_into_variant.children_variants
                    stop = children_variants and get_smart_step_into_variant_from_frame_offset(back.f_lasti, children_variants) is get_smart_step_into_variant_from_frame_offset(pydev_smart_child_offset, children_variants)
            if not stop:
                return
        if stop:
            py_db.set_suspend(thread_info.thread, step_cmd, original_step_cmd=info.pydev_original_step_cmd)
            _do_wait_suspend(py_db, thread_info, frame, 'line', None)
            return