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
def _claim_channel_id(self, channel_id):
    if channel_id in self._used_channel_ids:
        raise ConnectionError(f'Channel {channel_id!r} already open')
    else:
        self._used_channel_ids.append(channel_id)
        return channel_id