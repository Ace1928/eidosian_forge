import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _do_write(self):
    try:
        while self._write_backlog:
            data = self._write_backlog[0]
            count = self._sslobj.write(data)
            data_len = len(data)
            if count < data_len:
                self._write_backlog[0] = data[count:]
                self._write_buffer_size -= count
            else:
                del self._write_backlog[0]
                self._write_buffer_size -= data_len
    except SSLAgainErrors:
        pass
    self._process_outgoing()