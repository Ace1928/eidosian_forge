import errno
import os
import socket
import struct
import sys
import traceback
import warnings
from . import _auth
from .charset import charset_by_name, charset_by_id
from .constants import CLIENT, COMMAND, CR, ER, FIELD_TYPE, SERVER_STATUS
from . import converters
from .cursors import Cursor
from .optionfile import Parser
from .protocol import (
from . import err, VERSION_STRING
def _read_query_result(self, unbuffered=False):
    self._result = None
    if unbuffered:
        try:
            result = MySQLResult(self)
            result.init_unbuffered_query()
        except:
            result.unbuffered_active = False
            result.connection = None
            raise
    else:
        result = MySQLResult(self)
        result.read()
    self._result = result
    if result.server_status is not None:
        self.server_status = result.server_status
    return result.affected_rows