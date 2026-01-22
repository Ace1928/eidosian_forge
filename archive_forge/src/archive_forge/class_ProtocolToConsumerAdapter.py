import random
from typing import Any, Callable, Optional
from zope.interface import implementer
from twisted.internet import defer, error, interfaces
from twisted.internet.interfaces import IAddress, ITransport
from twisted.logger import _loggerFor
from twisted.python import components, failure, log
@implementer(interfaces.IConsumer)
class ProtocolToConsumerAdapter(components.Adapter):

    def write(self, data: bytes) -> None:
        self.original.dataReceived(data)

    def registerProducer(self, producer, streaming):
        pass

    def unregisterProducer(self):
        pass