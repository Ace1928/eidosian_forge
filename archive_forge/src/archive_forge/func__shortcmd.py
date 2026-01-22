import errno
import re
import socket
import sys
def _shortcmd(self, line):
    self._putcmd(line)
    return self._getresp()