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
class _ProtocolThreeEncoder:
    response_marker = request_marker = MESSAGE_VERSION_THREE
    BUFFER_SIZE = 1024 * 1024

    def __init__(self, write_func):
        self._buf = []
        self._buf_len = 0
        self._real_write_func = write_func

    def _write_func(self, bytes):
        self._buf.append(bytes)
        self._buf_len += len(bytes)
        if self._buf_len > self.BUFFER_SIZE:
            self.flush()

    def flush(self):
        if self._buf:
            self._real_write_func(b''.join(self._buf))
            del self._buf[:]
            self._buf_len = 0

    def _serialise_offsets(self, offsets):
        """Serialise a readv offset list."""
        txt = []
        for start, length in offsets:
            txt.append(b'%d,%d' % (start, length))
        return b'\n'.join(txt)

    def _write_protocol_version(self):
        self._write_func(MESSAGE_VERSION_THREE)

    def _write_prefixed_bencode(self, structure):
        bytes = bencode(structure)
        self._write_func(struct.pack('!L', len(bytes)))
        self._write_func(bytes)

    def _write_headers(self, headers):
        self._write_prefixed_bencode(headers)

    def _write_structure(self, args):
        self._write_func(b's')
        utf8_args = []
        for arg in args:
            if isinstance(arg, str):
                utf8_args.append(arg.encode('utf8'))
            else:
                utf8_args.append(arg)
        self._write_prefixed_bencode(utf8_args)

    def _write_end(self):
        self._write_func(b'e')
        self.flush()

    def _write_prefixed_body(self, bytes):
        self._write_func(b'b')
        self._write_func(struct.pack('!L', len(bytes)))
        self._write_func(bytes)

    def _write_chunked_body_start(self):
        self._write_func(b'oC')

    def _write_error_status(self):
        self._write_func(b'oE')

    def _write_success_status(self):
        self._write_func(b'oS')