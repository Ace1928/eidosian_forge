from binascii import hexlify
from collections import deque
import socket
import threading
import time
from paramiko.common import DEBUG, io_sleep
from paramiko.file import BufferedFile
from paramiko.util import u
from paramiko.sftp import (
from paramiko.sftp_attr import SFTPAttributes
def _start_prefetch(self, chunks, max_concurrent_requests=None):
    self._prefetching = True
    self._prefetch_done = False
    t = threading.Thread(target=self._prefetch_thread, args=(chunks, max_concurrent_requests))
    t.daemon = True
    t.start()