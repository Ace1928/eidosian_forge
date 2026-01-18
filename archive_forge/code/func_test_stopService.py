import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_stopService(self):
    """
        L{ProcessMonitor.stopService} should stop all monitored processes.
        """
    self.pm.addProcess('foo', ['foo'])
    self.pm.addProcess('bar', ['bar'])
    self.pm.startService()
    self.reactor.advance(self.pm.threshold)
    self.assertIn('foo', self.pm.protocols)
    self.assertIn('bar', self.pm.protocols)
    self.reactor.advance(1)
    self.pm.stopService()
    self.reactor.advance(self.pm.killTime + 1)
    self.assertEqual({}, self.pm.protocols)