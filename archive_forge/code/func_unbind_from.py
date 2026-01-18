from __future__ import annotations
import numbers
from .abstract import MaybeChannelBound, Object
from .exceptions import ContentDisallowed
from .serialization import prepare_accept_content
def unbind_from(self, exchange='', routing_key='', arguments=None, nowait=False, channel=None):
    """Unbind queue by deleting the binding from the server."""
    return (channel or self.channel).queue_unbind(queue=self.name, exchange=exchange.name, routing_key=routing_key, arguments=arguments, nowait=nowait)