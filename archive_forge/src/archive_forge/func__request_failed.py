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
def _request_failed(self, m):
    self.lock.acquire()
    try:
        msgs = self._close_internal()
    finally:
        self.lock.release()
    for m in msgs:
        if m is not None:
            self.transport._send_user_message(m)