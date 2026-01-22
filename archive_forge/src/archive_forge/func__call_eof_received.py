import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _call_eof_received(self):
    try:
        if self._app_state == AppProtocolState.STATE_CON_MADE:
            self._app_state = AppProtocolState.STATE_EOF
            keep_open = self._app_protocol.eof_received()
            if keep_open:
                logger.warning('returning true from eof_received() has no effect when using ssl')
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as ex:
        self._fatal_error(ex, 'Error calling eof_received()')