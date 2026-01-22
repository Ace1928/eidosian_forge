import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _check_handshake_timeout(self):
    if self._state == SSLProtocolState.DO_HANDSHAKE:
        msg = f'SSL handshake is taking longer than {self._ssl_handshake_timeout} seconds: aborting the connection'
        self._fatal_error(ConnectionAbortedError(msg))