import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _fatal_error(self, exc, message='Fatal error on transport'):
    if self._transport:
        self._transport._force_close(exc)
    if isinstance(exc, OSError):
        if self._loop.get_debug():
            logger.debug('%r: %s', self, message, exc_info=True)
    elif not isinstance(exc, exceptions.CancelledError):
        self._loop.call_exception_handler({'message': message, 'exception': exc, 'transport': self._transport, 'protocol': self})