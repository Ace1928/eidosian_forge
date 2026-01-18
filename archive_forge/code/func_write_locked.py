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
@contextlib.contextmanager
def write_locked(lockable):
    lockable.lock_write()
    try:
        yield lockable
    finally:
        lockable.unlock()