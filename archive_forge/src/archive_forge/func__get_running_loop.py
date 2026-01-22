import contextvars
import os
import socket
import subprocess
import sys
import threading
from . import format_helpers
def _get_running_loop():
    """Return the running event loop or None.

    This is a low-level function intended to be used by event loops.
    This function is thread-specific.
    """
    running_loop, pid = _running_loop.loop_pid
    if running_loop is not None and pid == os.getpid():
        return running_loop