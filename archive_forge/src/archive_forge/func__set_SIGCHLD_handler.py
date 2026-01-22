import signal
import threading
from . import IS_WINDOWS
from torch._C import _set_worker_pids, _remove_worker_pids  # noqa: F401
from torch._C import _error_if_any_worker_fails, _set_worker_signal_handlers  # noqa: F401
def _set_SIGCHLD_handler():
    if IS_WINDOWS:
        return
    if not isinstance(threading.current_thread(), threading._MainThread):
        return
    global _SIGCHLD_handler_set
    if _SIGCHLD_handler_set:
        return
    previous_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(previous_handler):
        previous_handler = None

    def handler(signum, frame):
        _error_if_any_worker_fails()
        if previous_handler is not None:
            assert callable(previous_handler)
            previous_handler(signum, frame)
    signal.signal(signal.SIGCHLD, handler)
    _SIGCHLD_handler_set = True