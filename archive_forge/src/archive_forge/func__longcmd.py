import errno
import re
import socket
import sys
def _longcmd(self, line):
    self._putcmd(line)
    return self._getlongresp()