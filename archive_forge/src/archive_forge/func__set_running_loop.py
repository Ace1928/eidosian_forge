import contextvars
import os
import socket
import subprocess
import sys
import threading
from . import format_helpers
def _set_running_loop(loop):
    """Set the running event loop.

    This is a low-level function intended to be used by event loops.
    This function is thread-specific.
    """
    _running_loop.loop_pid = (loop, os.getpid())