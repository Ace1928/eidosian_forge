from a file, a socket or a WSGI environment. The parser can be used to replace
import re
import sys
from io import BytesIO
from tempfile import TemporaryFile
from urllib.parse import parse_qs
from wsgiref.headers import Headers
from collections.abc import MutableMapping as DictMixin
def _iterparse(self):
    lines, line = (self._lineiter(), '')
    separator = b'--' + to_bytes(self.boundary)
    terminator = b'--' + to_bytes(self.boundary) + b'--'
    for line, nl in lines:
        if line in (separator, terminator):
            break
    else:
        raise MultipartError('Stream does not contain boundary')
    if line == terminator:
        for _ in lines:
            raise MultipartError('Data after end of stream')
        return
    mem_used, disk_used = (0, 0)
    is_tail = False
    opts = {'buffer_size': self.buffer_size, 'memfile_limit': self.memfile_limit, 'charset': self.charset}
    part = MultipartPart(**opts)
    for line, nl in lines:
        if line == terminator and (not is_tail):
            part.file.seek(0)
            yield part
            break
        elif line == separator and (not is_tail):
            if part.is_buffered():
                mem_used += part.size
            else:
                disk_used += part.size
            part.file.seek(0)
            yield part
            part = MultipartPart(**opts)
        else:
            is_tail = not nl
            try:
                part.feed(line, nl)
                if part.is_buffered():
                    if part.size + mem_used > self.mem_limit:
                        raise MultipartError('Memory limit reached.')
                elif part.size + disk_used > self.disk_limit:
                    raise MultipartError('Disk limit reached.')
            except MultipartError:
                part.close()
                raise
    else:
        part.close()
    if line != terminator:
        raise MultipartError('Unexpected end of multipart stream.')