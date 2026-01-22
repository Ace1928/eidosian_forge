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
def _update_chunk_length(self):
    if self.chunk_left is not None:
        return
    line = self._fp.fp.readline()
    line = line.split(b';', 1)[0]
    try:
        self.chunk_left = int(line, 16)
    except ValueError:
        self.close()
        raise InvalidChunkLength(self, line)