import itertools
import json
import linecache
import os
import platform
import sys
from functools import partial
import pydevd_file_utils
from _pydev_bundle import pydev_log
from _pydevd_bundle._debug_adapter import pydevd_base_schema, pydevd_schema
from _pydevd_bundle._debug_adapter.pydevd_schema import (
from _pydevd_bundle.pydevd_api import PyDevdAPI
from _pydevd_bundle.pydevd_breakpoints import get_exception_class, FunctionBreakpoint
from _pydevd_bundle.pydevd_comm_constants import (
from _pydevd_bundle.pydevd_filtering import ExcludeFilter
from _pydevd_bundle.pydevd_json_debug_options import _extract_debug_options, DebugOptions
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_utils import convert_dap_log_message_to_expression, ScopeRequest
from _pydevd_bundle.pydevd_constants import (PY_IMPL_NAME, DebugInfoHolder, PY_VERSION_STR,
from _pydevd_bundle.pydevd_trace_dispatch import USING_CYTHON
from _pydevd_frame_eval.pydevd_frame_eval_main import USING_FRAME_EVAL
from _pydevd_bundle.pydevd_comm import internal_get_step_in_targets_json
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id
def _create_breakpoint_from_add_breakpoint_result(self, py_db, source, breakpoint_id, result):
    error_code = result.error_code
    if error_code:
        if error_code == self.api.ADD_BREAKPOINT_FILE_NOT_FOUND:
            error_msg = 'Breakpoint in file that does not exist.'
        elif error_code == self.api.ADD_BREAKPOINT_FILE_EXCLUDED_BY_FILTERS:
            error_msg = 'Breakpoint in file excluded by filters.'
            if py_db.get_use_libraries_filter():
                error_msg += '\nNote: may be excluded because of "justMyCode" option (default == true).Try setting "justMyCode": false in the debug configuration (e.g., launch.json).\n'
        elif error_code == self.api.ADD_BREAKPOINT_LAZY_VALIDATION:
            error_msg = 'Waiting for code to be loaded to verify breakpoint.'
        elif error_code == self.api.ADD_BREAKPOINT_INVALID_LINE:
            error_msg = 'Breakpoint added to invalid line.'
        else:
            error_msg = 'Breakpoint not validated (reason unknown -- please report as bug).'
        return pydevd_schema.Breakpoint(verified=False, id=breakpoint_id, line=result.translated_line, message=error_msg, source=source).to_dict()
    else:
        return pydevd_schema.Breakpoint(verified=True, id=breakpoint_id, line=result.translated_line, source=source).to_dict()