from yowsup.structs import ProtocolTreeNode
import math
import binascii
import sys
import zlib
def unpackNibble(self, n):
    if n in range(0, 10):
        return n + 48
    if n in (10, 11):
        return 45 + (n - 10)
    raise ValueError('bad nibble %s' % n)