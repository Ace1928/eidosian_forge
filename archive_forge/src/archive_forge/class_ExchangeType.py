from __future__ import annotations
import re
from kombu.utils.text import escape_regex
class ExchangeType:
    """Base class for exchanges.

    Implements the specifics for an exchange type.

    Arguments:
    ---------
        channel (ChannelT): AMQ Channel.
    """
    type = None

    def __init__(self, channel):
        self.channel = channel

    def lookup(self, table, exchange, routing_key, default):
        """Lookup all queues matching `routing_key` in `exchange`.

        Returns
        -------
            str: queue name, or 'default' if no queues matched.
        """
        raise NotImplementedError('subclass responsibility')

    def prepare_bind(self, queue, exchange, routing_key, arguments):
        """Prepare queue-binding.

        Returns
        -------
            Tuple[str, Pattern, str]: of `(routing_key, regex, queue)`
                to be stored for bindings to this exchange.
        """
        return (routing_key, None, queue)

    def equivalent(self, prev, exchange, type, durable, auto_delete, arguments):
        """Return true if `prev` and `exchange` is equivalent."""
        return type == prev['type'] and durable == prev['durable'] and (auto_delete == prev['auto_delete']) and ((arguments or {}) == (prev['arguments'] or {}))