from __future__ import annotations
import os
import subprocess
import sys
import threading
import time
import debugpy
from debugpy import adapter
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import components, sessions
import traceback
import io
def wait_until_disconnected():
    """Blocks until all debug servers disconnect from the adapter.

    If there are no server connections, waits until at least one is established first,
    before waiting for it to disconnect.
    """
    while True:
        _connections_changed.wait()
        with _lock:
            _connections_changed.clear()
            if not len(_connections):
                return