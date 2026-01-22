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
def _make_legacy_ssl_transport(self, rawsock, protocol, sslcontext, waiter, *, server_side=False, server_hostname=None, extra=None, server=None):
    return _SelectorSslTransport(self, rawsock, protocol, sslcontext, waiter, server_side, server_hostname, extra, server)