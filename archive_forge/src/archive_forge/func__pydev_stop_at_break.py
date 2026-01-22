import sys
from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_comm import get_global_debugger
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame, NORM_PATHS_AND_BASE_CONTAINER
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
def _pydev_stop_at_break(line):
    frame = sys._getframe(1)
    t = threading.current_thread()
    try:
        additional_info = t.additional_info
    except:
        additional_info = set_additional_thread_info(t)
    if additional_info.is_tracing:
        return
    additional_info.is_tracing += 1
    try:
        py_db = get_global_debugger()
        if py_db is None:
            return
        pydev_log.debug('Setting f_trace due to frame eval mode in file: %s on line %s', frame.f_code.co_filename, line)
        additional_info.trace_suspend_type = 'frame_eval'
        pydevd_frame_eval_cython_wrapper = sys.modules['_pydevd_frame_eval.pydevd_frame_eval_cython_wrapper']
        thread_info = pydevd_frame_eval_cython_wrapper.get_thread_info_py()
        if thread_info.thread_trace_func is not None:
            frame.f_trace = thread_info.thread_trace_func
        else:
            frame.f_trace = py_db.get_thread_local_trace_func()
    finally:
        additional_info.is_tracing -= 1