import re
import itertools
class QRAlphaNum(QR):
    valid = re.compile(u'[-0-9A-Z $%*+./:]*$').match
    chars = u'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:'
    bits = (6, 5)
    group = 2
    mode = 2
    lengthbits = (9, 11, 13)