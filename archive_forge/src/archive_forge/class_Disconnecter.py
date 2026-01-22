import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
class Disconnecter(protocol.Protocol):
    """
            Protocol for the server side of the connection which disconnects
            itself in a callback on clientPaused and publishes notification
            when its connection is actually lost.
            """

    def connectionMade(self):
        """
                Set up a callback on clientPaused to lose the connection.
                """
        msg('Disconnector.connectionMade')

        def disconnect(ignored):
            msg('Disconnector.connectionMade disconnect')
            self.transport.loseConnection()
            msg('loseConnection called')
        clientPaused.addCallback(disconnect)

    def connectionLost(self, reason):
        """
                Notify observers that the server side of the connection has
                ended.
                """
        msg('Disconnecter.connectionLost')
        serverLost.callback(None)
        msg('serverLost called back')