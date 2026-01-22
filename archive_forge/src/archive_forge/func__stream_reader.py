import collections
import socket
import sys
import warnings
import weakref
from . import coroutines
from . import events
from . import exceptions
from . import format_helpers
from . import protocols
from .log import logger
from .tasks import sleep
@property
def _stream_reader(self):
    if self._stream_reader_wr is None:
        return None
    return self._stream_reader_wr()