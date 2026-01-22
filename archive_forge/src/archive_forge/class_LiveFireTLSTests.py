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
@skipIf(skipSSL, 'SSL not available')
@skipIf(reactorLacksSSL, 'This test case requires SSL support in the reactor')
class LiveFireTLSTests(LiveFireBase, TestCase):
    clientProto = SecurableProto
    serverProto = SecurableProto

    def test_liveFireCustomTLS(self):
        """
        Using real, live TLS, actually negotiate a connection.

        This also looks at the 'peerCertificate' attribute's correctness, since
        that's actually loaded using OpenSSL calls, but the main purpose is to
        make sure that we didn't miss anything obvious in iosim about TLS
        negotiations.
        """
        cert = tempcert
        self.svr.verifyFactory = lambda: [cert]
        self.svr.certFactory = lambda: cert

        def secured(rslt):
            x = cert.digest()

            def pinged(rslt2):
                self.assertEqual(x, self.cli.hostCertificate.digest())
                self.assertEqual(x, self.cli.peerCertificate.digest())
                self.assertEqual(x, self.svr.hostCertificate.digest())
                self.assertEqual(x, self.svr.peerCertificate.digest())
            return self.cli.callRemote(SecuredPing).addCallback(pinged)
        return self.cli.callRemote(amp.StartTLS, tls_localCertificate=cert, tls_verifyAuthorities=[cert]).addCallback(secured)