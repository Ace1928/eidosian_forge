import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
@property
def _protocol_paused(self):
    return self._ssl_protocol._app_writing_paused