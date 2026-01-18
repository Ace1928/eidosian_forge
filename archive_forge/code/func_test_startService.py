import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_startService(self):
    """
        L{ProcessMonitor.startService} starts all monitored processes.
        """
    self.pm.addProcess('foo', ['foo'])
    self.pm.startService()
    self.reactor.advance(0)
    self.assertIn('foo', self.pm.protocols)