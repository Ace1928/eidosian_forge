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
def _encode_bulk_data(self, body):
    """Encode body as a bulk data chunk."""
    return b''.join((b'%d\n' % len(body), body, b'done\n'))