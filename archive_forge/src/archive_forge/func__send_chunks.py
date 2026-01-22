import _thread
import struct
import sys
import time
from collections import deque
from io import BytesIO
from fastbencode import bdecode_as_tuple, bencode
import breezy
from ... import debug, errors, osutils
from ...trace import log_exception_quietly, mutter
from . import message, request
def _send_chunks(stream, write_func):
    for chunk in stream:
        if isinstance(chunk, bytes):
            data = ('%x\n' % len(chunk)).encode('ascii') + chunk
            write_func(data)
        elif isinstance(chunk, request.FailedSmartServerResponse):
            write_func(b'ERR\n')
            _send_chunks(chunk.args, write_func)
            return
        else:
            raise errors.BzrError('Chunks must be str or FailedSmartServerResponse, got %r' % chunk)