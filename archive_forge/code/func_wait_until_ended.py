import itertools
import os
import signal
import threading
import time
from debugpy import common
from debugpy.common import log, util
from debugpy.adapter import components, launchers, servers
def wait_until_ended():
    """Blocks until all sessions have ended.

    A session ends when all components that it manages disconnect from it.
    """
    while True:
        with _lock:
            if not len(_sessions):
                return
            _sessions_changed.clear()
        _sessions_changed.wait()