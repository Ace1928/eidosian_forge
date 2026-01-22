import errno
import socket
from zope.interface import verify
from twisted.internet.error import UnsupportedAddressFamily
from twisted.internet.interfaces import IReactorSocket
from twisted.internet.protocol import DatagramProtocol, ServerFactory
from twisted.internet.test.reactormixins import ReactorBuilder, needsRunningReactor
from twisted.python.log import err
from twisted.python.runtime import platform
class AdoptStreamConnectionErrorsTestsBuilder(ReactorBuilder):
    """
    Builder for testing L{IReactorSocket.adoptStreamConnection}
    implementations.

    Generally only tests for failure cases are found here.  Success cases for
    this interface are tested elsewhere.  For example, the success case for
    I{AF_INET} is in L{twisted.internet.test.test_tcp}, since that case should
    behave exactly the same as L{IReactorTCP.listenTCP}.
    """
    requiredInterfaces = [IReactorSocket]

    def test_invalidAddressFamily(self):
        """
        An implementation of L{IReactorSocket.adoptStreamConnection} raises
        L{UnsupportedAddressFamily} if passed an address family it does not
        support.
        """
        reactor = self.buildReactor()
        connection = socket.socket()
        self.addCleanup(connection.close)
        arbitrary = 2 ** 16 + 7
        self.assertRaises(UnsupportedAddressFamily, reactor.adoptStreamConnection, connection.fileno(), arbitrary, ServerFactory())