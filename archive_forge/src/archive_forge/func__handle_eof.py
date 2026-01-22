import binascii
import os
import socket
import time
import threading
from functools import wraps
from paramiko import util
from paramiko.common import (
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
from paramiko.file import BufferedFile
from paramiko.buffered_pipe import BufferedPipe, PipeTimeout
from paramiko import pipe
from paramiko.util import ClosingContextManager
def _handle_eof(self, m):
    self.lock.acquire()
    try:
        if not self.eof_received:
            self.eof_received = True
            self.in_buffer.close()
            self.in_stderr_buffer.close()
            if self._pipe is not None:
                self._pipe.set_forever()
    finally:
        self.lock.release()
    self._log(DEBUG, 'EOF received ({})'.format(self._name))