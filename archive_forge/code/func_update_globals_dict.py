import sys
from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_comm import get_global_debugger
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame, NORM_PATHS_AND_BASE_CONTAINER
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
def update_globals_dict(globals_dict):
    new_globals = {'_pydev_stop_at_break': _pydev_stop_at_break}
    globals_dict.update(new_globals)