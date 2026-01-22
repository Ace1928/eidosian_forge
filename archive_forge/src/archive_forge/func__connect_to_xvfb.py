import errno
import fcntl
import os
import subprocess
import time
from . import Connection, ConnectionException
def _connect_to_xvfb(self):
    for _ in range(100):
        try:
            conn = Connection(os.environ['DISPLAY'])
            conn.invalid()
            return conn
        except ConnectionException:
            time.sleep(0.2)
    assert False, "couldn't connect to xvfb"