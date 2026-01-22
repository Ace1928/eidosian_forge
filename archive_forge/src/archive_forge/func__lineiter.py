from a file, a socket or a WSGI environment. The parser can be used to replace
import re
import sys
from io import BytesIO
from tempfile import TemporaryFile
from urllib.parse import parse_qs
from wsgiref.headers import Headers
from collections.abc import MutableMapping as DictMixin
def _lineiter(self):
    """ Iterate over a binary file-like object line by line. Each line is
            returned as a (line, line_ending) tuple. If the line does not fit
            into self.buffer_size, line_ending is empty and the rest of the line
            is returned with the next iteration.
        """
    read = self.stream.read
    maxread, maxbuf = (self.content_length, self.buffer_size)
    buffer = b''
    while True:
        data = read(maxbuf if maxread < 0 else min(maxbuf, maxread))
        maxread -= len(data)
        lines = (buffer + data).splitlines(True)
        len_first_line = len(lines[0])
        if len_first_line > self.buffer_size:
            if len_first_line == self.buffer_size + 1 and lines[0].endswith(b'\r\n'):
                splitpos = self.buffer_size - 1
            else:
                splitpos = self.buffer_size
            lines[:1] = [lines[0][:splitpos], lines[0][splitpos:]]
        if data:
            buffer = lines[-1]
            lines = lines[:-1]
        for line in lines:
            if line.endswith(b'\r\n'):
                yield (line[:-2], b'\r\n')
            elif line.endswith(b'\n'):
                yield (line[:-1], b'\n')
            elif line.endswith(b'\r'):
                yield (line[:-1], b'\r')
            else:
                yield (line, b'')
        if not data:
            break