import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _test__append_write_backlog(self, data):
    self._ssl_protocol._write_backlog.append(data)
    self._ssl_protocol._write_buffer_size += len(data)