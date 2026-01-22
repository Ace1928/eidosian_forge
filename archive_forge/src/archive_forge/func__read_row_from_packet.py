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
def _read_row_from_packet(self, packet):
    row = []
    for encoding, converter in self.converters:
        try:
            data = packet.read_length_coded_string()
        except IndexError:
            break
        if data is not None:
            if encoding is not None:
                data = data.decode(encoding)
            if DEBUG:
                print('DEBUG: DATA = ', data)
            if converter is not None:
                data = converter(data)
        row.append(data)
    return tuple(row)