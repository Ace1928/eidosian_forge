import errno
import os
import selectors
import signal
import socket
import struct
import sys
import threading
import warnings
from . import connection
from . import process
from .context import reduction
from . import resource_tracker
from . import spawn
from . import util
def write_signed(fd, n):
    msg = SIGNED_STRUCT.pack(n)
    while msg:
        nbytes = os.write(fd, msg)
        if nbytes == 0:
            raise RuntimeError('should not get here')
        msg = msg[nbytes:]