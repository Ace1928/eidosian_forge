import email.parser
import email.message
import errno
import http
import io
import re
import socket
import sys
import collections.abc
from urllib.parse import urlsplit
def _read_chunked(self, amt=None):
    assert self.chunked != _UNKNOWN
    value = []
    try:
        while True:
            chunk_left = self._get_chunk_left()
            if chunk_left is None:
                break
            if amt is not None and amt <= chunk_left:
                value.append(self._safe_read(amt))
                self.chunk_left = chunk_left - amt
                break
            value.append(self._safe_read(chunk_left))
            if amt is not None:
                amt -= chunk_left
            self.chunk_left = 0
        return b''.join(value)
    except IncompleteRead as exc:
        raise IncompleteRead(b''.join(value)) from exc