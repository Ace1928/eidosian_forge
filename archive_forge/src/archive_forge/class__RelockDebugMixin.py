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
class _RelockDebugMixin:
    """Mixin support for -Drelock flag.

    Add this as a base class then call self._note_lock with 'r' or 'w' when
    acquiring a read- or write-lock.  If this object was previously locked (and
    locked the same way), and -Drelock is set, then this will trace.note a
    message about it.
    """
    _prev_lock = None

    def _note_lock(self, lock_type):
        if 'relock' in debug.debug_flags and self._prev_lock == lock_type:
            if lock_type == 'r':
                type_name = 'read'
            else:
                type_name = 'write'
            trace.note(gettext('{0!r} was {1} locked again'), self, type_name)
        self._prev_lock = lock_type