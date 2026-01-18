import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_stopServiceCancelRestarts(self):
    """
        L{ProcessMonitor.stopService} should cancel any scheduled process
        restarts.
        """
    self.pm.addProcess('foo', ['foo'])
    self.pm.startService()
    self.reactor.advance(self.pm.threshold)
    self.assertIn('foo', self.pm.protocols)
    self.reactor.advance(1)
    self.pm.protocols['foo'].processEnded(Failure(ProcessDone(0)))
    self.assertTrue(self.pm.restart['foo'].active())
    self.pm.stopService()
    self.assertFalse(self.pm.restart['foo'].active())