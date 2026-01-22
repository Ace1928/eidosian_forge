import collections
import errno
import functools
import selectors
import socket
import warnings
import weakref
from . import base_events
from . import constants
from . import events
from . import futures
from . import protocols
from . import sslproto
from . import transports
from . import trsock
from .log import logger
def _add_writer(self, fd, callback, *args):
    self._check_closed()
    handle = events.Handle(callback, args, self, None)
    try:
        key = self._selector.get_key(fd)
    except KeyError:
        self._selector.register(fd, selectors.EVENT_WRITE, (None, handle))
    else:
        mask, (reader, writer) = (key.events, key.data)
        self._selector.modify(fd, mask | selectors.EVENT_WRITE, (reader, handle))
        if writer is not None:
            writer.cancel()
    return handle