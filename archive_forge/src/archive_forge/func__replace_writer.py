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
def _replace_writer(self, writer):
    loop = self._loop
    transport = writer.transport
    self._stream_writer = writer
    self._transport = transport
    self._over_ssl = transport.get_extra_info('sslcontext') is not None