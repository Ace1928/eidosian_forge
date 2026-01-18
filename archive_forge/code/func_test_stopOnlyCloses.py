import errno
import socket
from zope.interface import verify
from twisted.internet.error import UnsupportedAddressFamily
from twisted.internet.interfaces import IReactorSocket
from twisted.internet.protocol import DatagramProtocol, ServerFactory
from twisted.internet.test.reactormixins import ReactorBuilder, needsRunningReactor
from twisted.python.log import err
from twisted.python.runtime import platform
def test_stopOnlyCloses(self):
    """
        When the L{IListeningPort} returned by
        L{IReactorSocket.adoptDatagramPort} is stopped using
        C{stopListening}, the underlying socket is closed but not
        shutdown.  This allows another process which still has a
        reference to it to continue reading and writing to it.
        """
    reactor = self.buildReactor()
    portSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.addCleanup(portSocket.close)
    portSocket.bind(('127.0.0.1', 0))
    portSocket.setblocking(False)
    port = reactor.adoptDatagramPort(portSocket.fileno(), portSocket.family, DatagramProtocol())
    d = port.stopListening()

    def stopped(ignored):
        exc = self.assertRaises(socket.error, portSocket.recvfrom, 1)
        if platform.isWindows():
            self.assertEqual(exc.args[0], errno.WSAEWOULDBLOCK)
        else:
            self.assertEqual(exc.args[0], errno.EAGAIN)
    d.addCallback(stopped)
    d.addErrback(err, 'Failed to read on original port.')
    needsRunningReactor(reactor, lambda: d.addCallback(lambda ignored: reactor.stop()))
    reactor.run()