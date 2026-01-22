import re
import itertools
class QRFNC1Second(QR):
    valid = re.compile('^([A-Za-z]|[0-9][0-9])$').match
    mode = 9
    lengthbits = (0, 0, 0)

    def write(self, buffer, version):
        self.write_header(buffer, version)
        d = self.data
        if len(d) == 1:
            d = ord(d) + 100
        else:
            d = int(d)
        buffer.put(d, 8)