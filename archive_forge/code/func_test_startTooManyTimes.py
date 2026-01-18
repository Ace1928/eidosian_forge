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
def test_startTooManyTimes(self):
    """
        Verify that the protocol will complain if we attempt to renegotiate TLS,
        which we don't support.
        """
    cli, svr, p = connectedServerAndClient(ServerClass=SecurableProto, ClientClass=SecurableProto)
    okc = OKCert()
    svr.certFactory = lambda: okc
    cli.callRemote(amp.StartTLS, tls_localCertificate=okc, tls_verifyAuthorities=[PretendRemoteCertificateAuthority()])
    p.flush()
    cli.noPeerCertificate = True
    self.assertRaises(amp.OnlyOneTLS, cli.callRemote, amp.StartTLS, tls_localCertificate=okc, tls_verifyAuthorities=[PretendRemoteCertificateAuthority()])