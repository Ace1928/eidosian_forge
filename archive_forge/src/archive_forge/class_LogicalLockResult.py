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
class LogicalLockResult:
    """The result of a lock_read/lock_write/lock_tree_write call on lockables.

    :ivar unlock: A callable which will unlock the lock.
    """

    def __init__(self, unlock, token=None):
        self.unlock = unlock
        self.token = token

    def __repr__(self):
        return 'LogicalLockResult(%s)' % self.unlock

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.unlock()
        except BaseException:
            if exc_type is None:
                raise
        return False