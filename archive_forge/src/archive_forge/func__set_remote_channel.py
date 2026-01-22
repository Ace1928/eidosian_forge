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
def _set_remote_channel(self, chanid, window_size, max_packet_size):
    self.remote_chanid = chanid
    self.out_window_size = window_size
    self.out_max_packet_size = self.transport._sanitize_packet_size(max_packet_size)
    self.active = 1
    self._log(DEBUG, 'Max packet out: {} bytes'.format(self.out_max_packet_size))