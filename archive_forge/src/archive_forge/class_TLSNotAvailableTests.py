import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
class TLSNotAvailableTests(TestCase):
    """
    Tests what happened when ssl is not available in current installation.
    """

    def setUp(self):
        """
        Disable ssl in amp.
        """
        self.ssl = amp.ssl
        amp.ssl = None

    def tearDown(self):
        """
        Restore ssl module.
        """
        amp.ssl = self.ssl

    def test_callRemoteError(self):
        """
        Check that callRemote raises an exception when called with a
        L{amp.StartTLS}.
        """
        cli, svr, p = connectedServerAndClient(ServerClass=SecurableProto, ClientClass=SecurableProto)
        okc = OKCert()
        svr.certFactory = lambda: okc
        return self.assertFailure(cli.callRemote(amp.StartTLS, tls_localCertificate=okc, tls_verifyAuthorities=[PretendRemoteCertificateAuthority()]), RuntimeError)

    def test_messageReceivedError(self):
        """
        When a client with SSL enabled talks to a server without SSL, it
        should return a meaningful error.
        """
        svr = SecurableProto()
        okc = OKCert()
        svr.certFactory = lambda: okc
        box = amp.Box()
        box[b'_command'] = b'StartTLS'
        box[b'_ask'] = b'1'
        boxes = []
        svr.sendBox = boxes.append
        svr.makeConnection(StringTransport())
        svr.ampBoxReceived(box)
        self.assertEqual(boxes, [{b'_error_code': b'TLS_ERROR', b'_error': b'1', b'_error_description': b'TLS not available'}])