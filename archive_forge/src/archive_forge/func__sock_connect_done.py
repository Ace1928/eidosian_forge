import collections
import errno
import functools
import socket
import sys
import warnings
from . import base_events
from . import constants
from . import events
from . import futures
from . import selectors
from . import transports
from . import sslproto
from .coroutines import coroutine
from .log import logger
def _sock_connect_done(self, fd, fut):
    self.remove_writer(fd)