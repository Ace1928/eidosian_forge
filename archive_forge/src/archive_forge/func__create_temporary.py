import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
def _create_temporary(path):
    """Create a temp file based on path and open for reading and writing."""
    return _create_carefully('%s.%s.%s.%s' % (path, int(time.time()), socket.gethostname(), os.getpid()))