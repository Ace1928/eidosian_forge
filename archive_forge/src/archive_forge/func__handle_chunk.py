from __future__ import absolute_import
import io
import logging
import zlib
from contextlib import contextmanager
from socket import error as SocketError
from socket import timeout as SocketTimeout
from ._collections import HTTPHeaderDict
from .connection import BaseSSLError, HTTPException
from .exceptions import (
from .packages import six
from .util.response import is_fp_closed, is_response_to_head
def _handle_chunk(self, amt):
    returned_chunk = None
    if amt is None:
        chunk = self._fp._safe_read(self.chunk_left)
        returned_chunk = chunk
        self._fp._safe_read(2)
        self.chunk_left = None
    elif amt < self.chunk_left:
        value = self._fp._safe_read(amt)
        self.chunk_left = self.chunk_left - amt
        returned_chunk = value
    elif amt == self.chunk_left:
        value = self._fp._safe_read(amt)
        self._fp._safe_read(2)
        self.chunk_left = None
        returned_chunk = value
    else:
        returned_chunk = self._fp._safe_read(self.chunk_left)
        self._fp._safe_read(2)
        self.chunk_left = None
    return returned_chunk