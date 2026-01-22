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
def _maybe_resume_transport(self):
    if self._paused and len(self._buffer) <= self._limit:
        self._paused = False
        self._transport.resume_reading()