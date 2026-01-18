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
def trace_exception(self, frame, event, arg):
    if event == 'exception':
        should_stop, frame = self._should_stop_on_exception(frame, event, arg)
        if should_stop:
            if self._handle_exception(frame, event, arg, EXCEPTION_TYPE_HANDLED):
                return self.trace_dispatch
    elif event == 'return':
        exc_info = self.exc_info
        if exc_info and arg is None:
            frame_skips_cache, frame_cache_key = (self._args[4], self._args[5])
            custom_key = (frame_cache_key, 'try_exc_info')
            container_obj = frame_skips_cache.get(custom_key)
            if container_obj is None:
                container_obj = frame_skips_cache[custom_key] = _TryExceptContainerObj()
            if is_unhandled_exception(container_obj, self._args[0], frame, exc_info[1], exc_info[2]) and self.handle_user_exception(frame):
                return self.trace_dispatch
    return self.trace_exception