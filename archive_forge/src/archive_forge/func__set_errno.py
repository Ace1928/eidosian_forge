import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def _set_errno(self, errno):
    self._backend.set_errno(errno)