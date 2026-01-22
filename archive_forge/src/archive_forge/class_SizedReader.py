import re
import sys
import tempfile
from urllib.parse import unquote
import cheroot.server
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
class SizedReader:

    def __init__(self, fp, length, maxbytes, bufsize=DEFAULT_BUFFER_SIZE, has_trailers=False):
        self.fp = fp
        self.length = length
        self.maxbytes = maxbytes
        self.buffer = b''
        self.bufsize = bufsize
        self.bytes_read = 0
        self.done = False
        self.has_trailers = has_trailers

    def read(self, size=None, fp_out=None):
        """Read bytes from the request body and return or write them to a file.

        A number of bytes less than or equal to the 'size' argument are read
        off the socket. The actual number of bytes read are tracked in
        self.bytes_read. The number may be smaller than 'size' when 1) the
        client sends fewer bytes, 2) the 'Content-Length' request header
        specifies fewer bytes than requested, or 3) the number of bytes read
        exceeds self.maxbytes (in which case, 413 is raised).

        If the 'fp_out' argument is None (the default), all bytes read are
        returned in a single byte string.

        If the 'fp_out' argument is not None, it must be a file-like
        object that supports the 'write' method; all bytes read will be
        written to the fp, and None is returned.
        """
        if self.length is None:
            if size is None:
                remaining = inf
            else:
                remaining = size
        else:
            remaining = self.length - self.bytes_read
            if size and size < remaining:
                remaining = size
        if remaining == 0:
            self.finish()
            if fp_out is None:
                return b''
            else:
                return None
        chunks = []
        if self.buffer:
            if remaining is inf:
                data = self.buffer
                self.buffer = b''
            else:
                data = self.buffer[:remaining]
                self.buffer = self.buffer[remaining:]
            datalen = len(data)
            remaining -= datalen
            self.bytes_read += datalen
            if self.maxbytes and self.bytes_read > self.maxbytes:
                raise cherrypy.HTTPError(413)
            if fp_out is None:
                chunks.append(data)
            else:
                fp_out.write(data)
        while remaining > 0:
            chunksize = min(remaining, self.bufsize)
            try:
                data = self.fp.read(chunksize)
            except Exception:
                e = sys.exc_info()[1]
                if e.__class__.__name__ == 'MaxSizeExceeded':
                    raise cherrypy.HTTPError(413, 'Maximum request length: %r' % e.args[1])
                else:
                    raise
            if not data:
                self.finish()
                break
            datalen = len(data)
            remaining -= datalen
            self.bytes_read += datalen
            if self.maxbytes and self.bytes_read > self.maxbytes:
                raise cherrypy.HTTPError(413)
            if fp_out is None:
                chunks.append(data)
            else:
                fp_out.write(data)
        if fp_out is None:
            return b''.join(chunks)

    def readline(self, size=None):
        """Read a line from the request body and return it."""
        chunks = []
        while size is None or size > 0:
            chunksize = self.bufsize
            if size is not None and size < self.bufsize:
                chunksize = size
            data = self.read(chunksize)
            if not data:
                break
            pos = data.find(b'\n') + 1
            if pos:
                chunks.append(data[:pos])
                remainder = data[pos:]
                self.buffer += remainder
                self.bytes_read -= len(remainder)
                break
            else:
                chunks.append(data)
        return b''.join(chunks)

    def readlines(self, sizehint=None):
        """Read lines from the request body and return them."""
        if self.length is not None:
            if sizehint is None:
                sizehint = self.length - self.bytes_read
            else:
                sizehint = min(sizehint, self.length - self.bytes_read)
        lines = []
        seen = 0
        while True:
            line = self.readline()
            if not line:
                break
            lines.append(line)
            seen += len(line)
            if seen >= sizehint:
                break
        return lines

    def finish(self):
        self.done = True
        if self.has_trailers and hasattr(self.fp, 'read_trailer_lines'):
            self.trailers = {}
            try:
                for line in self.fp.read_trailer_lines():
                    if line[0] in b' \t':
                        v = line.strip()
                    else:
                        try:
                            k, v = line.split(b':', 1)
                        except ValueError:
                            raise ValueError('Illegal header line.')
                        k = k.strip().title()
                        v = v.strip()
                    if k in cheroot.server.comma_separated_headers:
                        existing = self.trailers.get(k)
                        if existing:
                            v = b', '.join((existing, v))
                    self.trailers[k] = v
            except Exception:
                e = sys.exc_info()[1]
                if e.__class__.__name__ == 'MaxSizeExceeded':
                    raise cherrypy.HTTPError(413, 'Maximum request length: %r' % e.args[1])
                else:
                    raise