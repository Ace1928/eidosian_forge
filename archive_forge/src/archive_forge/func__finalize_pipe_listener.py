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
@staticmethod
def _finalize_pipe_listener(queue, address):
    util.sub_debug('closing listener with address=%r', address)
    for handle in queue:
        _winapi.CloseHandle(handle)