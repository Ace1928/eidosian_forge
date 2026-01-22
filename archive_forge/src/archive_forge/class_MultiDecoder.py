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
class MultiDecoder(object):
    """
    From RFC7231:
        If one or more encodings have been applied to a representation, the
        sender that applied the encodings MUST generate a Content-Encoding
        header field that lists the content codings in the order in which
        they were applied.
    """

    def __init__(self, modes):
        self._decoders = [_get_decoder(m.strip()) for m in modes.split(',')]

    def flush(self):
        return self._decoders[0].flush()

    def decompress(self, data):
        for d in reversed(self._decoders):
            data = d.decompress(data)
        return data