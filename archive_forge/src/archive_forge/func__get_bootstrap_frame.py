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
def _get_bootstrap_frame(depth: int) -> Tuple[Optional[FrameType], bool]:
    try:
        return (_thread_local_info.f_bootstrap, _thread_local_info.is_bootstrap_frame_internal)
    except:
        frame = _getframe(depth)
        f_bootstrap = frame
        is_bootstrap_frame_internal = False
        while f_bootstrap is not None:
            filename = f_bootstrap.f_code.co_filename
            name = splitext(basename(filename))[0]
            if name == 'threading':
                if f_bootstrap.f_code.co_name in ('__bootstrap', '_bootstrap'):
                    return (None, False)
                elif f_bootstrap.f_code.co_name in ('__bootstrap_inner', '_bootstrap_inner', 'is_alive'):
                    is_bootstrap_frame_internal = True
                    break
            elif name == 'pydev_monkey':
                if f_bootstrap.f_code.co_name == '__call__':
                    is_bootstrap_frame_internal = True
                    break
            elif name == 'pydevd':
                if f_bootstrap.f_code.co_name in ('run', 'main'):
                    return (None, False)
                if f_bootstrap.f_code.co_name == '_exec':
                    is_bootstrap_frame_internal = True
                    break
            elif f_bootstrap.f_back is None:
                break
            f_bootstrap = f_bootstrap.f_back
        if f_bootstrap is not None:
            _thread_local_info.is_bootstrap_frame_internal = is_bootstrap_frame_internal
            _thread_local_info.f_bootstrap = f_bootstrap
            return (_thread_local_info.f_bootstrap, _thread_local_info.is_bootstrap_frame_internal)
        return (f_bootstrap, is_bootstrap_frame_internal)