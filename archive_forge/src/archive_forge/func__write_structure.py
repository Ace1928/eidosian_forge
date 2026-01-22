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
def _write_structure(self, args):
    self._write_func(b's')
    utf8_args = []
    for arg in args:
        if isinstance(arg, str):
            utf8_args.append(arg.encode('utf8'))
        else:
            utf8_args.append(arg)
    self._write_prefixed_bencode(utf8_args)