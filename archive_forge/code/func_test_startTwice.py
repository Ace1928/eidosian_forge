import pickle
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.application import internet
from twisted.application.internet import (
from twisted.internet import task
from twisted.internet.defer import CancelledError, Deferred
from twisted.internet.interfaces import (
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.logger import formatEvent, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_startTwice(self):
    """
        If L{ClientService} is started when it's already started, it will log a
        complaint and do nothing else (in particular it will not make
        additional connections).
        """
    cq, service = self.makeReconnector(fireImmediately=False, startService=False)
    self.assertEqual(len(cq.connectQueue), 0)
    service.startService()
    self.assertEqual(len(cq.connectQueue), 1)
    messages = catchLogs(self)
    service.startService()
    self.assertEqual(len(cq.connectQueue), 1)
    self.assertIn('Duplicate ClientService.startService', messages()[0])