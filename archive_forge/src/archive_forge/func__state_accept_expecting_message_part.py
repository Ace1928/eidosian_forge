import _thread
import struct
import sys
import time
from collections import deque
from io import BytesIO
from fastbencode import bdecode_as_tuple, bencode
import breezy
from ... import debug, errors, osutils
from ...trace import log_exception_quietly, mutter
from . import message, request
def _state_accept_expecting_message_part(self):
    message_part_kind = self._extract_single_byte()
    if message_part_kind == b'o':
        self.state_accept = self._state_accept_expecting_one_byte
    elif message_part_kind == b's':
        self.state_accept = self._state_accept_expecting_structure
    elif message_part_kind == b'b':
        self.state_accept = self._state_accept_expecting_bytes
    elif message_part_kind == b'e':
        self.done()
    else:
        raise errors.SmartProtocolError('Bad message kind byte: {!r}'.format(message_part_kind))