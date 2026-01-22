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
def _prefetch_thread(self, chunks, max_concurrent_requests):
    for offset, length in chunks:
        if max_concurrent_requests is not None:
            while True:
                with self._prefetch_lock:
                    pf_len = len(self._prefetch_extents)
                    if pf_len < max_concurrent_requests:
                        break
                time.sleep(io_sleep)
        num = self.sftp._async_request(self, CMD_READ, self.handle, int64(offset), int(length))
        with self._prefetch_lock:
            self._prefetch_extents[num] = (offset, length)