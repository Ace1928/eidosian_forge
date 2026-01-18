from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_constants import thread_get_ident, IS_CPYTHON, NULL
import ctypes
import time
from _pydev_bundle import pydev_log
import weakref
from _pydevd_bundle.pydevd_utils import is_current_thread_main_thread
from _pydevd_bundle import pydevd_utils

        This can be called regularly to always execute the given function after a given timeout:

        call_on_timeout(py_db, 10, on_timeout)


        Or as a context manager to stop the method from being called if it finishes before the timeout
        elapses:

        with call_on_timeout(py_db, 10, on_timeout):
            ...

        Note: the callback will be called from a PyDBDaemonThread.
        