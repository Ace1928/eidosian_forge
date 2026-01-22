import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _start_handshake(self):
    if self._loop.get_debug():
        logger.debug('%r starts SSL handshake', self)
        self._handshake_start_time = self._loop.time()
    else:
        self._handshake_start_time = None
    self._set_state(SSLProtocolState.DO_HANDSHAKE)
    self._handshake_timeout_handle = self._loop.call_later(self._ssl_handshake_timeout, lambda: self._check_handshake_timeout())
    self._do_handshake()