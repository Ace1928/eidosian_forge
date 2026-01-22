import errno
import os
import sys
from stat import S_IMODE, S_ISDIR, ST_MODE
from .. import osutils, transport, urlutils
def _pump_to_fd(self, fromfile, to_fd):
    """Copy contents of one file to another."""
    BUFSIZE = 32768
    while True:
        b = fromfile.read(BUFSIZE)
        if not b:
            break
        os.write(to_fd, b)