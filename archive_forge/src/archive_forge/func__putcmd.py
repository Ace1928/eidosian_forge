import errno
import re
import socket
import sys
def _putcmd(self, line):
    if self._debugging:
        print('*cmd*', repr(line))
    line = bytes(line, self.encoding)
    self._putline(line)