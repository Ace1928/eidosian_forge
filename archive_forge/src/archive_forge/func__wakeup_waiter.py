import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _wakeup_waiter(self, exc=None):
    if self._waiter is None:
        return
    if not self._waiter.cancelled():
        if exc is not None:
            self._waiter.set_exception(exc)
        else:
            self._waiter.set_result(None)
    self._waiter = None