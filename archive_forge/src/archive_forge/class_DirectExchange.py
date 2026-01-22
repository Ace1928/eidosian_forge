from __future__ import annotations
import re
from kombu.utils.text import escape_regex
class DirectExchange(ExchangeType):
    """Direct exchange.

    The `direct` exchange routes based on exact routing keys.
    """
    type = 'direct'

    def lookup(self, table, exchange, routing_key, default):
        return {queue for rkey, _, queue in table if rkey == routing_key}

    def deliver(self, message, exchange, routing_key, **kwargs):
        _lookup = self.channel._lookup
        _put = self.channel._put
        for queue in _lookup(exchange, routing_key):
            _put(queue, message, **kwargs)