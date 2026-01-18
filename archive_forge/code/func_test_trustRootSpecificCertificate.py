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
def test_trustRootSpecificCertificate(self):
    """
        Specifying a L{Certificate} object for L{trustRoot} will result in that
        certificate being the only trust root for a client.
        """
    caCert, serverCert = certificatesForAuthorityAndServer()
    otherCa, otherServer = certificatesForAuthorityAndServer()
    sProto, cProto, sWrapped, cWrapped, pump = loopbackTLSConnection(trustRoot=caCert, privateKeyFile=pathContainingDumpOf(self, serverCert.privateKey), chainedCertFile=pathContainingDumpOf(self, serverCert))
    pump.flush()
    self.assertIsNone(cWrapped.lostReason)
    self.assertEqual(cWrapped.data, sWrapped.greeting)