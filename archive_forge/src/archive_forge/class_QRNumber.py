import re
import itertools
class QRNumber(QR):
    valid = re.compile(u'[0-9]*$').match
    chars = u'0123456789'
    bits = (4, 3, 3)
    group = 3
    mode = 1
    lengthbits = (10, 12, 14)