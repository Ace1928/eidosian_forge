from paste.util import intset
import socket
def _int2ip(self, num):
    rv = []
    for i in range(4):
        rv.append(str(num & 255))
        num >>= 8
    return '.'.join(reversed(rv))