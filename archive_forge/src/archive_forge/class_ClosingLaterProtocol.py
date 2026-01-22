import socket
from gc import collect
from typing import Optional
from weakref import ref
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.interfaces import IConnector, IReactorFDSet
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.reactormixins import needsRunningReactor
from twisted.python import context, log
from twisted.python.failure import Failure
from twisted.python.log import ILogContext, err, msg
from twisted.python.runtime import platform
from twisted.test.test_tcp import ClosingProtocol
from twisted.trial.unittest import SkipTest
class ClosingLaterProtocol(ConnectableProtocol):
    """
    ClosingLaterProtocol exchanges one byte with its peer and then disconnects
    itself.  This is mostly a work-around for the fact that connectionMade is
    called before the SSL handshake has completed.
    """

    def __init__(self, onConnectionLost):
        self.lostConnectionReason = None
        self.onConnectionLost = onConnectionLost

    def connectionMade(self):
        msg('ClosingLaterProtocol.connectionMade')

    def dataReceived(self, bytes):
        msg(f'ClosingLaterProtocol.dataReceived {bytes!r}')
        self.transport.loseConnection()

    def connectionLost(self, reason):
        msg('ClosingLaterProtocol.connectionLost')
        self.lostConnectionReason = reason
        self.onConnectionLost.callback(self)