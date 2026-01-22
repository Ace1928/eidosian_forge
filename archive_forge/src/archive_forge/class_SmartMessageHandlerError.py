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
class SmartMessageHandlerError(errors.InternalBzrError):
    _fmt = 'The message handler raised an exception:\n%(traceback_text)s'

    def __init__(self, exc_info):
        import traceback
        self.exc_type, self.exc_value, self.exc_tb = exc_info
        self.exc_info = exc_info
        traceback_strings = traceback.format_exception(self.exc_type, self.exc_value, self.exc_tb)
        self.traceback_text = ''.join(traceback_strings)