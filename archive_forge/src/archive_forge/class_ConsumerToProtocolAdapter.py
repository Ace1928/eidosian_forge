import random
from typing import Any, Callable, Optional
from zope.interface import implementer
from twisted.internet import defer, error, interfaces
from twisted.internet.interfaces import IAddress, ITransport
from twisted.logger import _loggerFor
from twisted.python import components, failure, log
@implementer(interfaces.IProtocol)
class ConsumerToProtocolAdapter(components.Adapter):

    def dataReceived(self, data: bytes) -> None:
        self.original.write(data)

    def connectionLost(self, reason: failure.Failure) -> None:
        pass

    def makeConnection(self, transport):
        pass

    def connectionMade(self):
        pass