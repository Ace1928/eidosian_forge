import re
import itertools
class QR8bitByte(QR):
    bits = (8,)
    group = 1
    mode = 4
    lengthbits = (8, 16, 16)

    def __init__(self, data):
        if isinstance(data, unicode):
            self.data = data.encode('utf-8')
        else:
            self.data = data

    def write(self, buffer, version):
        self.write_header(buffer, version)
        for c in self.data:
            if isinstance(c, str):
                c = ord(c)
            buffer.put(c, 8)