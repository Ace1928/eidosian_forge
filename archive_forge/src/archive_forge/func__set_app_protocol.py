import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _set_app_protocol(self, app_protocol):
    self._app_protocol = app_protocol
    if hasattr(app_protocol, 'get_buffer') and isinstance(app_protocol, protocols.BufferedProtocol):
        self._app_protocol_get_buffer = app_protocol.get_buffer
        self._app_protocol_buffer_updated = app_protocol.buffer_updated
        self._app_protocol_is_buffer = True
    else:
        self._app_protocol_is_buffer = False