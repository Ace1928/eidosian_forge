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
def _remove_writer(self, fd):
    """Remove a writer callback."""
    if self.is_closed():
        return False
    try:
        key = self._selector.get_key(fd)
    except KeyError:
        return False
    else:
        mask, (reader, writer) = (key.events, key.data)
        mask &= ~selectors.EVENT_WRITE
        if not mask:
            self._selector.unregister(fd)
        else:
            self._selector.modify(fd, mask, (reader, None))
        if writer is not None:
            writer.cancel()
            return True
        else:
            return False