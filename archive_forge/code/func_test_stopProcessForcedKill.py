import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_stopProcessForcedKill(self):
    """
        L{ProcessMonitor.stopProcess} kills a process which fails to terminate
        naturally within L{ProcessMonitor.killTime} seconds.
        """
    self.pm.startService()
    self.pm.addProcess('foo', ['foo'])
    self.assertIn('foo', self.pm.protocols)
    self.reactor.advance(self.pm.threshold)
    proc = self.pm.protocols['foo'].transport
    proc._terminationDelay = self.pm.killTime + 1
    self.pm.stopProcess('foo')
    self.reactor.advance(self.pm.killTime - 1)
    self.assertEqual(0.0, self.pm.timeStarted['foo'])
    self.reactor.advance(1)
    self.reactor.pump([0, 0])
    self.assertEqual(self.reactor.seconds(), self.pm.timeStarted['foo'])