from _pydevd_bundle.pydevd_constants import get_current_thread_id, Null, ForkSafeLock
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
from _pydev_bundle._pydev_saved_modules import thread, threading
import sys
from _pydev_bundle import pydev_log

    It's possible to show paused frames by adding a custom frame through this API (it's
    intended to be used for coroutines, but could potentially be used for generators too).

    :param frame:
        The topmost frame to be shown paused when a thread with thread.ident == thread_id is paused.

    :param name:
        The name to be shown for the custom thread in the UI.

    :param thread_id:
        The thread id to which this frame is related (must match thread.ident).

    :return: str
        Returns the custom thread id which will be used to show the given frame paused.
    