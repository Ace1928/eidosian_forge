import select
import socket
import sys
import time
import warnings
import os
from errno import EALREADY, EINPROGRESS, EWOULDBLOCK, ECONNRESET, EINVAL, \
class file_dispatcher(dispatcher):

    def __init__(self, fd, map=None):
        dispatcher.__init__(self, None, map)
        self.connected = True
        try:
            fd = fd.fileno()
        except AttributeError:
            pass
        self.set_file(fd)
        os.set_blocking(fd, False)

    def set_file(self, fd):
        self.socket = file_wrapper(fd)
        self._fileno = self.socket.fileno()
        self.add_channel()