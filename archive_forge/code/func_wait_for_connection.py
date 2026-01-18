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
def wait_for_connection(session, predicate, timeout=None):
    """Waits until there is a server matching the specified predicate connected to
    this adapter, and returns the corresponding Connection.

    If there is more than one server connection already available, returns the oldest
    one.
    """

    def wait_for_timeout():
        time.sleep(timeout)
        wait_for_timeout.timed_out = True
        with _lock:
            _connections_changed.set()
    wait_for_timeout.timed_out = timeout == 0
    if timeout:
        thread = threading.Thread(target=wait_for_timeout, name='servers.wait_for_connection() timeout')
        thread.daemon = True
        thread.start()
    if timeout != 0:
        log.info('{0} waiting for connection from debug server...', session)
    while True:
        with _lock:
            _connections_changed.clear()
            conns = (conn for conn in _connections if predicate(conn))
            conn = next(conns, None)
            if conn is not None or wait_for_timeout.timed_out:
                return conn
        _connections_changed.wait()