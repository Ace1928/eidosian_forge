import collections
import threading
import time
import socket
import warnings
import queue
from jaraco.functools import pass_none
def _clear_threads(self):
    """Clear self._threads and yield all joinable threads."""
    threads, self._threads[:] = (self._threads[:], [])
    return (thread for thread in threads if thread is not threading.current_thread())