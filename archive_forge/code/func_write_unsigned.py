import errno
import os
import selectors
import signal
import socket
import struct
import sys
import threading
from . import connection
from . import process
from . import reduction
from . import semaphore_tracker
from . import spawn
from . import util
from .compat import spawnv_passfds
def write_unsigned(fd, n):
    msg = UNSIGNED_STRUCT.pack(n)
    while msg:
        nbytes = os.write(fd, msg)
        if nbytes == 0:
            raise RuntimeError('should not get here')
        msg = msg[nbytes:]