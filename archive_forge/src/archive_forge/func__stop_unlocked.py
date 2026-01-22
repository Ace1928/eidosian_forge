import errno
import os
import selectors
import signal
import socket
import struct
import sys
import threading
import warnings
from . import connection
from . import process
from .context import reduction
from . import resource_tracker
from . import spawn
from . import util
def _stop_unlocked(self):
    if self._forkserver_pid is None:
        return
    os.close(self._forkserver_alive_fd)
    self._forkserver_alive_fd = None
    os.waitpid(self._forkserver_pid, 0)
    self._forkserver_pid = None
    if not util.is_abstract_socket_namespace(self._forkserver_address):
        os.unlink(self._forkserver_address)
    self._forkserver_address = None