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
def _on_tune(self, channel_max, frame_max, server_heartbeat, argsig='BlB'):
    client_heartbeat = self.client_heartbeat or 0
    self.channel_max = channel_max or self.channel_max
    self.frame_max = frame_max or self.frame_max
    self.server_heartbeat = server_heartbeat or 0
    if self.server_heartbeat == 0 or client_heartbeat == 0:
        self.heartbeat = max(self.server_heartbeat, client_heartbeat)
    else:
        self.heartbeat = min(self.server_heartbeat, client_heartbeat)
    if not self.client_heartbeat:
        self.heartbeat = 0
    self.send_method(spec.Connection.TuneOk, argsig, (self.channel_max, self.frame_max, self.heartbeat), callback=self._on_tune_sent)