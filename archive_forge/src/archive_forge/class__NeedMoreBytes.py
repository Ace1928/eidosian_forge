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
class _NeedMoreBytes(Exception):
    """Raise this inside a _StatefulDecoder to stop decoding until more bytes
    have been received.
    """

    def __init__(self, count=None):
        """Constructor.

        :param count: the total number of bytes needed by the current state.
            May be None if the number of bytes needed is unknown.
        """
        self.count = count