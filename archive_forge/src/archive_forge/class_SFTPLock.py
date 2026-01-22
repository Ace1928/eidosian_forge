import bisect
import errno
import itertools
import os
import random
import stat
import sys
import time
import warnings
from .. import config, debug, errors, urlutils
from ..errors import LockError, ParamikoNotPresent, PathError, TransportError
from ..osutils import fancy_rename
from ..trace import mutter, warning
from ..transport import (ConnectedTransport, FileExists, FileFileStream,
class SFTPLock:
    """This fakes a lock in a remote location.

    A present lock is indicated just by the existence of a file.  This
    doesn't work well on all transports and they are only used in
    deprecated storage formats.
    """
    __slots__ = ['path', 'lock_path', 'lock_file', 'transport']

    def __init__(self, path, transport):
        self.lock_file = None
        self.path = path
        self.lock_path = path + '.write-lock'
        self.transport = transport
        try:
            abspath = transport._remote_path(self.lock_path)
            self.lock_file = transport._sftp_open_exclusive(abspath)
        except FileExists:
            raise LockError('File {!r} already locked'.format(self.path))

    def unlock(self):
        if not self.lock_file:
            return
        self.lock_file.close()
        self.lock_file = None
        try:
            self.transport.delete(self.lock_path)
        except (NoSuchFile,):
            pass