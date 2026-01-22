import errno
import sys
import time
from array import array
from socket import AF_INET, AF_INET6, SOCK_STREAM, SOL_SOCKET, socket
from struct import pack
from unittest import skipIf
from zope.interface.verify import verifyClass
from twisted.internet.interfaces import IPushProducer
from twisted.python.log import msg
from twisted.trial.unittest import TestCase
class SupportTests(TestCase):
    """
    Tests for L{twisted.internet.iocpreactor.iocpsupport}, low-level reactor
    implementation helpers.
    """

    def _acceptAddressTest(self, family, localhost):
        """
        Create a C{SOCK_STREAM} connection to localhost using a socket with an
        address family of C{family} and assert that the result of
        L{iocpsupport.get_accept_addrs} is consistent with the result of
        C{socket.getsockname} and C{socket.getpeername}.

        A port starts listening (is bound) at the low-level socket without
        calling accept() yet.
        A client is then connected.
        After the client is connected IOCP accept() is called, which is the
        target of these tests.

        Most of the time, the socket is ready instantly, but sometimes
        the socket is not ready right away after calling IOCP accept().
        It should not take more than 5 seconds for a socket to be ready, as
        the client connection is already made over the loopback interface.

        These are flaky tests.
        Tweak the failure rate by changing the number of retries and the
        wait/sleep between retries.

        If you will need to update the retries to wait more than 5 seconds
        for the port to be available, then there might a bug in the code and
        not the test (or a very, very busy VM running the tests).
        """
        msg(f'family = {family!r}')
        port = socket(family, SOCK_STREAM)
        self.addCleanup(port.close)
        port.bind(('', 0))
        port.listen(1)
        client = socket(family, SOCK_STREAM)
        self.addCleanup(client.close)
        client.setblocking(False)
        try:
            client.connect((localhost, port.getsockname()[1]))
        except OSError as e:
            self.assertIn(e.errno, (errno.EINPROGRESS, errno.EWOULDBLOCK))
        server = socket(family, SOCK_STREAM)
        self.addCleanup(server.close)
        buff = array('B', b'\x00' * 256)
        self.assertEqual(0, _iocp.accept(port.fileno(), server.fileno(), buff, None))
        for attemptsRemaining in reversed(range(5)):
            try:
                server.setsockopt(SOL_SOCKET, SO_UPDATE_ACCEPT_CONTEXT, pack('P', port.fileno()))
                break
            except OSError as socketError:
                if socketError.errno != getattr(errno, 'WSAENOTCONN'):
                    raise
                if attemptsRemaining == 0:
                    raise
            time.sleep(0.2)
        self.assertEqual((family, client.getpeername()[:2], client.getsockname()[:2]), _iocp.get_accept_addrs(server.fileno(), buff))

    def test_ipv4AcceptAddress(self):
        """
        L{iocpsupport.get_accept_addrs} returns a three-tuple of address
        information about the socket associated with the file descriptor passed
        to it.  For a connection using IPv4:

          - the first element is C{AF_INET}
          - the second element is a two-tuple of a dotted decimal notation IPv4
            address and a port number giving the peer address of the connection
          - the third element is the same type giving the host address of the
            connection
        """
        self._acceptAddressTest(AF_INET, '127.0.0.1')

    @skipIf(ipv6Skip, ipv6SkipReason)
    def test_ipv6AcceptAddress(self):
        """
        Like L{test_ipv4AcceptAddress}, but for IPv6 connections.
        In this case:

          - the first element is C{AF_INET6}
          - the second element is a two-tuple of a hexadecimal IPv6 address
            literal and a port number giving the peer address of the connection
          - the third element is the same type giving the host address of the
            connection
        """
        self._acceptAddressTest(AF_INET6, '::1')