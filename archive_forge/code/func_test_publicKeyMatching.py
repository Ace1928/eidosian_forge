import datetime
import itertools
import sys
from unittest import skipIf
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.error import CertificateError, ConnectionClosed, ConnectionLost
from twisted.internet.task import Clock
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.reflect import requireModule
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_twisted import SetAsideModule
from twisted.trial import util
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
def test_publicKeyMatching(self):
    """
        L{PublicKey.matches} returns L{True} for keys from certificates with
        the same key, and L{False} for keys from certificates with different
        keys.
        """
    hostA = sslverify.Certificate.loadPEM(A_HOST_CERTIFICATE_PEM)
    hostB = sslverify.Certificate.loadPEM(A_HOST_CERTIFICATE_PEM)
    peerA = sslverify.Certificate.loadPEM(A_PEER_CERTIFICATE_PEM)
    self.assertTrue(hostA.getPublicKey().matches(hostB.getPublicKey()))
    self.assertFalse(peerA.getPublicKey().matches(hostA.getPublicKey()))