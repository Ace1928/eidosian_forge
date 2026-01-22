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
def _wait_for_event(self):
    self.event.wait()
    assert self.event.is_set()
    if self.event_ready:
        return
    e = self.transport.get_exception()
    if e is None:
        e = SSHException('Channel closed.')
    raise e