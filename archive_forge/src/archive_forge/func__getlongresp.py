import errno
import re
import socket
import sys
def _getlongresp(self):
    resp = self._getresp()
    list = []
    octets = 0
    line, o = self._getline()
    while line != b'.':
        if line.startswith(b'..'):
            o = o - 1
            line = line[1:]
        octets = octets + o
        list.append(line)
        line, o = self._getline()
    return (resp, list, octets)