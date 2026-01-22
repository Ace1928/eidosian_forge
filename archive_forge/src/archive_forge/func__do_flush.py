import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _do_flush(self):
    self._do_read()
    self._set_state(SSLProtocolState.SHUTDOWN)
    self._do_shutdown()