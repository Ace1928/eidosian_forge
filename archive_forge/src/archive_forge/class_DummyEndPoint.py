from __future__ import annotations
from zope.interface import implementer, verify
from twisted.internet import defer, interfaces
from twisted.trial import unittest
from twisted.web import client
@implementer(interfaces.IStreamClientEndpoint)
class DummyEndPoint:
    """An endpoint that does not connect anywhere"""

    def __init__(self, someString: str) -> None:
        self.someString = someString

    def __repr__(self) -> str:
        return f'DummyEndPoint({self.someString})'

    def connect(self, factory: interfaces.IProtocolFactory) -> defer.Deferred[dict[str, interfaces.IProtocolFactory]]:
        return defer.succeed(dict(factory=factory))