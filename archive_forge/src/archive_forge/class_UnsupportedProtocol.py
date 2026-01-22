import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
class UnsupportedProtocol(errors.PathError):
    _fmt = 'Unsupported protocol for url "%(path)s"%(extra)s'

    def __init__(self, url, extra=''):
        errors.PathError.__init__(self, url, extra=extra)