import contextlib
import errno
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple
from . import debug, errors, osutils, trace
from .hooks import Hooks
from .i18n import gettext
from .transport import Transport
class _OSLock:

    def __init__(self):
        self.f = None
        self.filename = None

    def _open(self, filename, filemode):
        self.filename = osutils.realpath(filename)
        try:
            self.f = open(self.filename, filemode)
            return self.f
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EPERM):
                raise errors.LockFailed(self.filename, str(e))
            if e.errno != errno.ENOENT:
                raise
            trace.mutter('trying to create missing lock %r', self.filename)
            self.f = open(self.filename, 'wb+')
            return self.f

    def _clear_f(self):
        """Clear the self.f attribute cleanly."""
        if self.f:
            self.f.close()
            self.f = None

    def unlock(self):
        raise NotImplementedError()