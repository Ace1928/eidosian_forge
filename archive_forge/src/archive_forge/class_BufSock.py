import io
from oslotest import base
from oslo_privsep import comm
class BufSock(object):

    def __init__(self):
        self.readpos = 0
        self.buf = io.BytesIO()

    def recv(self, bufsize):
        if self.buf.closed:
            return b''
        self.buf.seek(self.readpos, 0)
        data = self.buf.read(bufsize)
        self.readpos += len(data)
        return data

    def sendall(self, data):
        self.buf.seek(0, 2)
        self.buf.write(data)

    def shutdown(self, _flag):
        self.buf.close()