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
class ContextFactory:

    def getContext(self):
        """
            Create a context for the server side of the connection.

            @return: an SSL context using a certificate and key.
            @rtype: C{OpenSSL.SSL.Context}
            """
        ctx = SSL.Context(SSL.SSLv23_METHOD)
        if chainedCertFile is not None:
            ctx.use_certificate_chain_file(chainedCertFile)
        ctx.use_privatekey_file(privateKeyFile)
        ctx.check_privatekey()
        return ctx