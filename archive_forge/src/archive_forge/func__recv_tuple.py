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
def _recv_tuple(self):
    """Receive a tuple from the medium request."""
    return _decode_tuple(self._request.read_line())