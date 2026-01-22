import io
import os
import sys
import socket
import struct
import time
import tempfile
import itertools
from . import util
from . import AuthenticationError, BufferTooShort
from .context import reduction
def _bad_message_length(self):
    if self._writable:
        self._readable = False
    else:
        self.close()
    raise OSError('bad message length')