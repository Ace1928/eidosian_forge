from __future__ import annotations
from itertools import count
from typing import TYPE_CHECKING
from .common import maybe_declare
from .compression import compress
from .connection import is_connection, maybe_channel
from .entity import Exchange, Queue, maybe_delivery_mode
from .exceptions import ContentDisallowed
from .serialization import dumps, prepare_accept_content
from .utils.functional import ChannelPromise, maybe_list
def _basic_consume(self, queue, consumer_tag=None, no_ack=no_ack, nowait=True):
    tag = self._active_tags.get(queue.name)
    if tag is None:
        tag = self._add_tag(queue, consumer_tag)
        queue.consume(tag, self._receive_callback, no_ack=no_ack, nowait=nowait)
    return tag