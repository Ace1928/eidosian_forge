from _pydevd_bundle.pydevd_constants import get_current_thread_id, Null, ForkSafeLock
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
from _pydev_bundle._pydev_saved_modules import thread, threading
import sys
from _pydev_bundle import pydev_log
class CustomFrame:

    def __init__(self, name, frame, thread_id):
        self.name = name
        self.frame = frame
        self.mod_time = 0
        self.thread_id = thread_id