import codecs
import errno
import fcntl
import io
import os
import pty
import resource
import signal
import struct
import sys
import termios
import time
from pty import (STDIN_FILENO, CHILD)
from .util import which, PtyProcessError
def _writeb(self, b, flush=True):
    n = self.fileobj.write(b)
    if flush:
        self.fileobj.flush()
    return n