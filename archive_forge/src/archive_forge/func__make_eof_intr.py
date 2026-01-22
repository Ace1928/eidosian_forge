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
def _make_eof_intr():
    """Set constants _EOF and _INTR.
    
    This avoids doing potentially costly operations on module load.
    """
    global _EOF, _INTR
    if _EOF is not None and _INTR is not None:
        return
    try:
        from termios import VEOF, VINTR
        fd = None
        for name in ('stdin', 'stdout'):
            stream = getattr(sys, '__%s__' % name, None)
            if stream is None or not hasattr(stream, 'fileno'):
                continue
            try:
                fd = stream.fileno()
            except ValueError:
                continue
        if fd is None:
            raise ValueError('No stream has a fileno')
        intr = ord(termios.tcgetattr(fd)[6][VINTR])
        eof = ord(termios.tcgetattr(fd)[6][VEOF])
    except (ImportError, OSError, IOError, ValueError, termios.error):
        try:
            from termios import CEOF, CINTR
            intr, eof = (CINTR, CEOF)
        except ImportError:
            intr, eof = (3, 4)
    _INTR = _byte(intr)
    _EOF = _byte(eof)