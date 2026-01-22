import base64
import binascii
import collections
import html
from django.conf import settings
from django.core.exceptions import (
from django.core.files.uploadhandler import SkipFile, StopFutureHandlers, StopUpload
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.http import parse_header_parameters
from django.utils.regex_helper import _lazy_re_compile
class BoundaryIter:
    """
    A Producer that is sensitive to boundaries.

    Will happily yield bytes until a boundary is found. Will yield the bytes
    before the boundary, throw away the boundary bytes themselves, and push the
    post-boundary bytes back on the stream.

    The future calls to next() after locating the boundary will raise a
    StopIteration exception.
    """

    def __init__(self, stream, boundary):
        self._stream = stream
        self._boundary = boundary
        self._done = False
        self._rollback = len(boundary) + 6
        unused_char = self._stream.read(1)
        if not unused_char:
            raise InputStreamExhausted()
        self._stream.unget(unused_char)

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration()
        stream = self._stream
        rollback = self._rollback
        bytes_read = 0
        chunks = []
        for bytes in stream:
            bytes_read += len(bytes)
            chunks.append(bytes)
            if bytes_read > rollback:
                break
            if not bytes:
                break
        else:
            self._done = True
        if not chunks:
            raise StopIteration()
        chunk = b''.join(chunks)
        boundary = self._find_boundary(chunk)
        if boundary:
            end, next = boundary
            stream.unget(chunk[next:])
            self._done = True
            return chunk[:end]
        elif not chunk[:-rollback]:
            self._done = True
            return chunk
        else:
            stream.unget(chunk[-rollback:])
            return chunk[:-rollback]

    def _find_boundary(self, data):
        """
        Find a multipart boundary in data.

        Should no boundary exist in the data, return None. Otherwise, return
        a tuple containing the indices of the following:
         * the end of current encapsulation
         * the start of the next encapsulation
        """
        index = data.find(self._boundary)
        if index < 0:
            return None
        else:
            end = index
            next = index + len(self._boundary)
            last = max(0, end - 1)
            if data[last:last + 1] == b'\n':
                end -= 1
            last = max(0, end - 1)
            if data[last:last + 1] == b'\r':
                end -= 1
            return (end, next)