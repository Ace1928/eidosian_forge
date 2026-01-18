import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_stopServiceCleanupScheduledRestarts(self):
    """
        L{ProcessMonitor.stopService} should cancel all scheduled process
        restarts.
        """
    self.pm.threshold = 5
    self.pm.minRestartDelay = 5
    self.pm.startService()
    self.pm.addProcess('foo', ['foo'])
    self.reactor.advance(1)
    self.pm.stopProcess('foo')
    self.reactor.advance(1)
    self.pm.stopService()
    self.reactor.advance(6)
    self.assertEqual(self.pm.protocols, {})