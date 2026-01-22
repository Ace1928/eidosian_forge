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
def _on_blocked(self):
    """Callback called when connection blocked.

        Notes:
            This is an RabbitMQ Extension.
        """
    reason = 'connection blocked, see broker logs'
    if self.on_blocked:
        return self.on_blocked(reason)