import io
import os
class Unreader(object):

    def __init__(self):
        self.buf = io.BytesIO()

    def chunk(self):
        raise NotImplementedError()

    def read(self, size=None):
        if size is not None and (not isinstance(size, int)):
            raise TypeError('size parameter must be an int or long.')
        if size is not None:
            if size == 0:
                return b''
            if size < 0:
                size = None
        self.buf.seek(0, os.SEEK_END)
        if size is None and self.buf.tell():
            ret = self.buf.getvalue()
            self.buf = io.BytesIO()
            return ret
        if size is None:
            d = self.chunk()
            return d
        while self.buf.tell() < size:
            chunk = self.chunk()
            if not chunk:
                ret = self.buf.getvalue()
                self.buf = io.BytesIO()
                return ret
            self.buf.write(chunk)
        data = self.buf.getvalue()
        self.buf = io.BytesIO()
        self.buf.write(data[size:])
        return data[:size]

    def unread(self, data):
        self.buf.seek(0, os.SEEK_END)
        self.buf.write(data)