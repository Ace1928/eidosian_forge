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
def _delivery_details(self, exchange, delivery_mode=None, maybe_delivery_mode=maybe_delivery_mode, Exchange=Exchange):
    if isinstance(exchange, Exchange):
        return (exchange.name, maybe_delivery_mode(delivery_mode or exchange.delivery_mode))
    return (exchange, maybe_delivery_mode(delivery_mode or self.exchange.delivery_mode))