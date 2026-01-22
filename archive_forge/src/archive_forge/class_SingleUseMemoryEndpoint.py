import os.path
from errno import ENOSYS
from struct import pack
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
import hamcrest
from twisted.conch.error import ConchError, HostKeyChanged, UserRejectedKey
from twisted.conch.interfaces import IConchUser
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.portal import Portal
from twisted.internet.address import IPv4Address
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import (
from twisted.internet.interfaces import IAddress, IStreamClientEndpoint
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import (
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
from twisted.test.iosim import FakeTransport, connect
@implementer(IStreamClientEndpoint)
class SingleUseMemoryEndpoint:
    """
    L{SingleUseMemoryEndpoint} is a client endpoint which allows one connection
    to be set up and then exposes an API for moving around bytes related to
    that connection.

    @ivar pump: L{None} until a connection is attempted, then a L{IOPump}
        instance associated with the protocol which is connected.
    @type pump: L{IOPump}
    """

    def __init__(self, server):
        """
        @param server: An L{IProtocol} provider to which the client will be
            connected.
        @type server: L{IProtocol} provider
        """
        self.pump = None
        self._server = server

    def connect(self, factory):
        if self.pump is not None:
            raise Exception('SingleUseMemoryEndpoint was already used')
        try:
            protocol = factory.buildProtocol(MemoryAddress())
        except BaseException:
            return fail()
        else:
            self.pump = connect(self._server, AbortableFakeTransport(self._server, isServer=True), protocol, AbortableFakeTransport(protocol, isServer=False))
            return succeed(protocol)