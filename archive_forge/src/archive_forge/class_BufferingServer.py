from io import BytesIO
from twisted.internet import abstract, defer, protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
class BufferingServer(protocol.Protocol):
    buffer = b''

    def dataReceived(self, data: bytes) -> None:
        self.buffer += data