import logging
import socket
import uuid
import warnings
from array import array
from time import monotonic
from vine import ensure_promise
from . import __version__, sasl, spec
from .abstract_channel import AbstractChannel
from .channel import Channel
from .exceptions import (AMQPDeprecationWarning, ChannelError, ConnectionError,
from .method_framing import frame_handler, frame_writer
from .transport import Transport
def Transport(self, host, connect_timeout, ssl=False, read_timeout=None, write_timeout=None, socket_settings=None, **kwargs):
    return Transport(host, connect_timeout=connect_timeout, ssl=ssl, read_timeout=read_timeout, write_timeout=write_timeout, socket_settings=socket_settings, **kwargs)