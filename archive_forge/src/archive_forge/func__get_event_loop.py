import contextvars
import os
import socket
import subprocess
import sys
import threading
from . import format_helpers
def _get_event_loop(stacklevel=3):
    current_loop = _get_running_loop()
    if current_loop is not None:
        return current_loop
    return get_event_loop_policy().get_event_loop()