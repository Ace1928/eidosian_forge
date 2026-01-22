import collections
import threading
import time
import socket
import warnings
import queue
from jaraco.functools import pass_none
def _clear_dead_threads(self):
    for t in [t for t in self._threads if not t.is_alive()]:
        self._threads.remove(t)
        try:
            self._pending_shutdowns.popleft()
        except IndexError:
            pass