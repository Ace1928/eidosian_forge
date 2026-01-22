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
def _data_in_prefetch_buffers(self, offset):
    """
        if a block of data is present in the prefetch buffers, at the given
        offset, return the offset of the relevant prefetch buffer.  otherwise,
        return None.  this guarantees nothing about the number of bytes
        collected in the prefetch buffer so far.
        """
    k = [i for i in self._prefetch_data.keys() if i <= offset]
    if len(k) == 0:
        return None
    index = max(k)
    buf_offset = offset - index
    if buf_offset >= len(self._prefetch_data[index]):
        return None
    return index