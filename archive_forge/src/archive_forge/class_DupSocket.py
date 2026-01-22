import os
import signal
import socket
import sys
import threading
from . import process
from .context import reduction
from . import util
class DupSocket(object):
    """Picklable wrapper for a socket."""

    def __init__(self, sock):
        new_sock = sock.dup()

        def send(conn, pid):
            share = new_sock.share(pid)
            conn.send_bytes(share)
        self._id = _resource_sharer.register(send, new_sock.close)

    def detach(self):
        """Get the socket.  This should only be called once."""
        with _resource_sharer.get_connection(self._id) as conn:
            share = conn.recv_bytes()
            return socket.fromshare(share)