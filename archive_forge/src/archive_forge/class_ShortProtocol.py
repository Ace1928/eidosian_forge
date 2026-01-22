from typing import Optional, Sequence, Type
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.endpoints import (
from twisted.internet.error import ConnectionClosed
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
class ShortProtocol(Protocol):

    def connectionMade(self):
        if not ITLSTransport.providedBy(self.transport):
            finished = self.factory.finished
            self.factory.finished = None
            finished.errback(SkipTest('No ITLSTransport support'))
            return
        self.transport.startTLS(self.factory.context)
        self.transport.write(b'x')

    def dataReceived(self, data):
        self.transport.write(b'y')
        self.transport.loseConnection()

    def connectionLost(self, reason):
        finished = self.factory.finished
        if finished is not None:
            self.factory.finished = None
            finished.callback(reason)