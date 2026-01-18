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
def write_packet(self, payload):
    """Writes an entire "mysql packet" in its entirety to the network
        adding its length and sequence number.
        """
    data = _pack_int24(len(payload)) + bytes([self._next_seq_id]) + payload
    if DEBUG:
        dump_packet(data)
    self._write_bytes(data)
    self._next_seq_id = (self._next_seq_id + 1) % 256