import errno
import socket
from pathlib import Path
from threading import Thread
import zmq
from jupyter_client.localinterfaces import localhost
def _bind_socket(self):
    try:
        win_in_use = errno.WSAEADDRINUSE
    except AttributeError:
        win_in_use = None
    max_attempts = 1 if self.original_port else 100
    for attempt in range(max_attempts):
        try:
            self._try_bind_socket()
        except zmq.ZMQError as ze:
            if attempt == max_attempts - 1:
                raise
            if ze.errno != errno.EADDRINUSE and ze.errno != win_in_use:
                raise
            if self.original_port == 0:
                self.pick_port()
            else:
                raise
        else:
            return