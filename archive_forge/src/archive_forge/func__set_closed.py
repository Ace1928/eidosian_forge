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
def _set_closed(self):
    self.closed = True
    self.in_buffer.close()
    self.in_stderr_buffer.close()
    self.out_buffer_cv.notify_all()
    self.event.set()
    self.status_event.set()
    if self._pipe is not None:
        self._pipe.set_forever()