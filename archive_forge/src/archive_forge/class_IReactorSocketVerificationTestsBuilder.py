import errno
import socket
from zope.interface import verify
from twisted.internet.error import UnsupportedAddressFamily
from twisted.internet.interfaces import IReactorSocket
from twisted.internet.protocol import DatagramProtocol, ServerFactory
from twisted.internet.test.reactormixins import ReactorBuilder, needsRunningReactor
from twisted.python.log import err
from twisted.python.runtime import platform
class IReactorSocketVerificationTestsBuilder(ReactorBuilder):
    """
    Builder for testing L{IReactorSocket} implementations for required
    methods and method signatures.

    L{ReactorBuilder} already runs L{IReactorSocket.providedBy} to
    ensure that these tests will only be run on reactor classes that
    claim to implement L{IReactorSocket}.

    These tests ensure that reactors which claim to provide the
    L{IReactorSocket} interface actually have all the required methods
    and that those methods have the expected number of arguments.

    These tests will be skipped for reactors which do not claim to
    provide L{IReactorSocket}.
    """
    requiredInterfaces = [IReactorSocket]

    def test_provider(self):
        """
        The reactor instance returned by C{buildReactor} provides
        L{IReactorSocket}.
        """
        reactor = self.buildReactor()
        self.assertTrue(verify.verifyObject(IReactorSocket, reactor))