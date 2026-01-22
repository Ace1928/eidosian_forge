from __future__ import annotations
import numbers
from .abstract import MaybeChannelBound, Object
from .exceptions import ContentDisallowed
from .serialization import prepare_accept_content
class binding(Object):
    """Represents a queue or exchange binding.

    Arguments:
    ---------
        exchange (Exchange): Exchange to bind to.
        routing_key (str): Routing key used as binding key.
        arguments (Dict): Arguments for bind operation.
        unbind_arguments (Dict): Arguments for unbind operation.
    """
    attrs = (('exchange', None), ('routing_key', None), ('arguments', None), ('unbind_arguments', None))

    def __init__(self, exchange=None, routing_key='', arguments=None, unbind_arguments=None):
        self.exchange = exchange
        self.routing_key = routing_key
        self.arguments = arguments
        self.unbind_arguments = unbind_arguments

    def declare(self, channel, nowait=False):
        """Declare destination exchange."""
        if self.exchange and self.exchange.name:
            self.exchange.declare(channel=channel, nowait=nowait)

    def bind(self, entity, nowait=False, channel=None):
        """Bind entity to this binding."""
        entity.bind_to(exchange=self.exchange, routing_key=self.routing_key, arguments=self.arguments, nowait=nowait, channel=channel)

    def unbind(self, entity, nowait=False, channel=None):
        """Unbind entity from this binding."""
        entity.unbind_from(self.exchange, routing_key=self.routing_key, arguments=self.unbind_arguments, nowait=nowait, channel=channel)

    def __repr__(self):
        return f'<binding: {self}>'

    def __str__(self):
        return '{}->{}'.format(_reprstr(self.exchange.name), _reprstr(self.routing_key))