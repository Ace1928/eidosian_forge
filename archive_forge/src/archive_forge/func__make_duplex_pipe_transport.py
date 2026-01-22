import io
import os
import socket
import warnings
import signal
import threading
import collections
from . import base_events
from . import constants
from . import futures
from . import exceptions
from . import protocols
from . import sslproto
from . import transports
from . import trsock
from .log import logger
def _make_duplex_pipe_transport(self, sock, protocol, waiter=None, extra=None):
    return _ProactorDuplexPipeTransport(self, sock, protocol, waiter, extra)