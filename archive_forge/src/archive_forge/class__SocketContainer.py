import errno
import io
import os
import sys
import socket
import select
import struct
import tempfile
import itertools
from . import reduction
from . import util
from . import AuthenticationError, BufferTooShort
from ._ext import _billiard
from .compat import setblocking, send_offset
from time import monotonic
from .reduction import ForkingPickler
class _SocketContainer:

    def __init__(self, sock):
        self.sock = sock