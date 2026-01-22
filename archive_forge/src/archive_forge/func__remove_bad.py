from __future__ import annotations
import errno
import math
import select as __select__
import sys
from numbers import Integral
from . import fileno
from .compat import detect_environment
def _remove_bad(self):
    for fd in self._rfd | self._wfd | self._efd:
        try:
            _selectf([fd], [], [], 0)
        except (_selecterr, OSError) as exc:
            if getattr(exc, 'errno', None) in SELECT_BAD_FD:
                self.unregister(fd)