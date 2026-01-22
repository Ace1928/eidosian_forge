import linecache
import os.path
import re
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_dont_trace
from _pydevd_bundle.pydevd_constants import (RETURN_VALUES_DICT, NO_FTRACE,
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, just_raised, remove_exception_from_frame, ignore_exception_trace
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
from _pydevd_bundle.pydevd_comm_constants import constant_to_str, CMD_SET_FUNCTION_BREAK
import sys
import dis
def _get_unfiltered_back_frame(self, main_debugger, frame):
    f = frame.f_back
    while f is not None:
        if not main_debugger.is_files_filter_enabled:
            return f
        elif main_debugger.apply_files_filter(f, f.f_code.co_filename, False):
            f = f.f_back
        else:
            return f
    return f