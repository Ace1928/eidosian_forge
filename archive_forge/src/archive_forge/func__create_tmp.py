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
def _create_tmp(self):
    """Create a file in the tmp subdirectory and open and return it."""
    now = time.time()
    hostname = socket.gethostname()
    if '/' in hostname:
        hostname = hostname.replace('/', '\\057')
    if ':' in hostname:
        hostname = hostname.replace(':', '\\072')
    uniq = '%s.M%sP%sQ%s.%s' % (int(now), int(now % 1 * 1000000.0), os.getpid(), Maildir._count, hostname)
    path = os.path.join(self._path, 'tmp', uniq)
    try:
        os.stat(path)
    except FileNotFoundError:
        Maildir._count += 1
        try:
            return _create_carefully(path)
        except FileExistsError:
            pass
    raise ExternalClashError('Name clash prevented file creation: %s' % path)