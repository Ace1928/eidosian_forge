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
def _translate_io_exception(self, e, path, more_info='', failure_exc=PathError):
    """Translate a paramiko or IOError into a friendlier exception.

        :param e: The original exception
        :param path: The path in question when the error is raised
        :param more_info: Extra information that can be included,
                          such as what was going on
        :param failure_exc: Paramiko has the super fun ability to raise completely
                           opaque errors that just set "e.args = ('Failure',)" with
                           no more information.
                           If this parameter is set, it defines the exception
                           to raise in these cases.
        """
    self._translate_error(e, path, raise_generic=False)
    if getattr(e, 'args', None) is not None:
        if e.args == ('No such file or directory',) or e.args == ('No such file',):
            raise NoSuchFile(path, str(e) + more_info)
        if e.args == ('mkdir failed',) or e.args[0].startswith('syserr: File exists'):
            raise FileExists(path, str(e) + more_info)
        if e.args == ('Failure',):
            raise failure_exc(path, str(e) + more_info)
        if e.args[0].startswith('Directory not empty: ') or getattr(e, 'errno', None) == errno.ENOTEMPTY:
            raise errors.DirectoryNotEmpty(path, str(e))
        if e.args == ('Operation unsupported',):
            raise errors.TransportNotPossible()
        mutter('Raising exception with args %s', e.args)
    if getattr(e, 'errno', None) is not None:
        mutter('Raising exception with errno %s', e.errno)
    raise e