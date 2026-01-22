import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _check_shutdown_timeout(self):
    if self._state in (SSLProtocolState.FLUSHING, SSLProtocolState.SHUTDOWN):
        self._transport._force_close(exceptions.TimeoutError('SSL shutdown timed out'))