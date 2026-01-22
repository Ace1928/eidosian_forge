import os
import random
import stat
import time
from io import BytesIO
from urllib.parse import urlparse, urlunparse
from .. import config, debug, errors, osutils, ui, urlutils
from ..tests.test_server import TestServer
from ..trace import mutter
from . import (ConnectedTransport, FileExists, FileStream, NoSuchFile,
def _translate_gio_error(self, err, path, extra=None):
    if 'gio' in debug.debug_flags:
        mutter('GIO Error: {} {}'.format(str(err), path))
    if extra is None:
        extra = str(err)
    if err.code == gio.ERROR_NOT_FOUND:
        raise NoSuchFile(path, extra=extra)
    elif err.code == gio.ERROR_EXISTS:
        raise FileExists(path, extra=extra)
    elif err.code == gio.ERROR_NOT_DIRECTORY:
        raise errors.NotADirectory(path, extra=extra)
    elif err.code == gio.ERROR_NOT_EMPTY:
        raise errors.DirectoryNotEmpty(path, extra=extra)
    elif err.code == gio.ERROR_BUSY:
        raise errors.ResourceBusy(path, extra=extra)
    elif err.code == gio.ERROR_PERMISSION_DENIED:
        raise errors.PermissionDenied(path, extra=extra)
    elif err.code == gio.ERROR_HOST_NOT_FOUND:
        raise errors.PathError(path, extra=extra)
    elif err.code == gio.ERROR_IS_DIRECTORY:
        raise errors.PathError(path, extra=extra)
    else:
        mutter('unable to understand error for path: %s: %s', path, err)
        raise errors.PathError(path, extra='Unhandled gio error: ' + str(err))