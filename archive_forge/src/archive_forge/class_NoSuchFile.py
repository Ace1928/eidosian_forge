import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
class NoSuchFile(errors.PathError):
    _fmt = 'No such file: %(path)r%(extra)s'