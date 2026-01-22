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
def _read_prefetch(self, size):
    """
        read data out of the prefetch buffer, if possible.  if the data isn't
        in the buffer, return None.  otherwise, behaves like a normal read.
        """
    while True:
        offset = self._data_in_prefetch_buffers(self._realpos)
        if offset is not None:
            break
        if self._prefetch_done or self._closed:
            break
        self.sftp._read_response()
        self._check_exception()
    if offset is None:
        self._prefetching = False
        return None
    prefetch = self._prefetch_data[offset]
    del self._prefetch_data[offset]
    buf_offset = self._realpos - offset
    if buf_offset > 0:
        self._prefetch_data[offset] = prefetch[:buf_offset]
        prefetch = prefetch[buf_offset:]
    if size < len(prefetch):
        self._prefetch_data[self._realpos + size] = prefetch[size:]
        prefetch = prefetch[:size]
    return prefetch