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
def test_passwordAuthenticationFailure(self):
    """
        If the SSH server rejects the password presented during authentication,
        the L{Deferred} returned by L{SSHCommandClientEndpoint.connect} fires
        with a L{Failure} wrapping L{AuthenticationFailed}.
        """
    endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', b'dummy user', self.hostname, self.port, password=b'dummy password', knownHosts=self.knownHosts, ui=FixedResponseUI(False))
    factory = Factory()
    factory.protocol = Protocol
    connected = endpoint.connect(factory)
    server, client, pump = self.connectedServerAndClient(self.factory, self.reactor.tcpClients[0][2])
    self.reactor.advance(server.service.passwordDelay)
    pump.flush()
    f = self.failureResultOf(connected)
    f.trap(AuthenticationFailed)
    self.assertClientTransportState(client, False)