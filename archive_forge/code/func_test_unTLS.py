import os
import hamcrest
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import waitUntilAllDisconnected
from twisted.protocols import basic
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test.test_tcp import ProperlyCloseFilesMixin
from twisted.trial.unittest import TestCase
from zope.interface import implementer
def test_unTLS(self):
    """
        Test for server startTLS not followed by a startTLS in client: the data
        received after server startTLS should be received as raw.
        """

    def check(ignored):
        self.assertEqual(self.serverFactory.lines, UnintelligentProtocol.pretext)
        self.assertTrue(self.serverFactory.rawdata, 'No encrypted bytes received')
    d = self._runTest(UnintelligentProtocol(), LineCollector(False, self.fillBuffer))
    return d.addCallback(check)