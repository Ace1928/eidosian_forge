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
def test_trustRootPlatformRejectsUntrustedCA(self):
    """
        Specifying a C{trustRoot} of L{platformTrust} when initializing
        L{sslverify.OpenSSLCertificateOptions} causes certificates issued by a
        newly created CA to be rejected by an SSL connection using these
        options.

        Note that this test should I{always} pass, even on platforms where the
        CA certificates are not installed, as long as L{platformTrust} rejects
        completely invalid / unknown root CA certificates.  This is simply a
        smoke test to make sure that verification is happening at all.
        """
    caSelfCert, serverCert = certificatesForAuthorityAndServer()
    chainedCert = pathContainingDumpOf(self, serverCert, caSelfCert)
    privateKey = pathContainingDumpOf(self, serverCert.privateKey)
    sProto, cProto, sWrapped, cWrapped, pump = loopbackTLSConnection(trustRoot=platformTrust(), privateKeyFile=privateKey, chainedCertFile=chainedCert)
    self.assertEqual(cWrapped.data, b'')
    self.assertEqual(cWrapped.lostReason.type, SSL.Error)
    err = cWrapped.lostReason.value
    self.assertEqual(err.args[0][0][2], 'tlsv1 alert unknown ca')