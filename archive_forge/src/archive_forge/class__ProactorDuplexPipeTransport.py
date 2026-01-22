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
class _ProactorDuplexPipeTransport(_ProactorReadPipeTransport, _ProactorBaseWritePipeTransport, transports.Transport):
    """Transport for duplex pipes."""

    def can_write_eof(self):
        return False

    def write_eof(self):
        raise NotImplementedError