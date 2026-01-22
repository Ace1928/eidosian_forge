from __future__ import annotations
import datetime
import errno
import socket
import struct
import time
from typing import (
from bson import _decode_all_selective
from pymongo import _csot, helpers, message, ssl_support
from pymongo.common import MAX_MESSAGE_SIZE
from pymongo.compression_support import _NO_COMPRESSION, decompress
from pymongo.errors import (
from pymongo.message import _UNPACK_REPLY, _OpMsg, _OpReply
from pymongo.monitoring import _is_speculative_authenticate
from pymongo.socket_checker import _errno_from_exception
def _receive_data_on_socket(conn: Connection, length: int, deadline: Optional[float]) -> memoryview:
    buf = bytearray(length)
    mv = memoryview(buf)
    bytes_read = 0
    while bytes_read < length:
        try:
            wait_for_read(conn, deadline)
            if _csot.get_timeout() and deadline is not None:
                conn.set_conn_timeout(max(deadline - time.monotonic(), 0))
            chunk_length = conn.conn.recv_into(mv[bytes_read:])
        except BLOCKING_IO_ERRORS:
            raise socket.timeout('timed out') from None
        except OSError as exc:
            if _errno_from_exception(exc) == errno.EINTR:
                continue
            raise
        if chunk_length == 0:
            raise OSError('connection closed')
        bytes_read += chunk_length
    return mv