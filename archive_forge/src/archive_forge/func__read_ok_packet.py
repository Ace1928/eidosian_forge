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
def _read_ok_packet(self, first_packet):
    ok_packet = OKPacketWrapper(first_packet)
    self.affected_rows = ok_packet.affected_rows
    self.insert_id = ok_packet.insert_id
    self.server_status = ok_packet.server_status
    self.warning_count = ok_packet.warning_count
    self.message = ok_packet.message
    self.has_next = ok_packet.has_next