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
def _add_reader(self, fd, callback, *args):
    if not self.is_reading():
        return
    self._loop._add_reader(fd, callback, *args)