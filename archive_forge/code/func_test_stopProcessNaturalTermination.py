import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_stopProcessNaturalTermination(self):
    """
        L{ProcessMonitor.stopProcess} immediately sends a TERM signal to the
        named process.
        """
    self.pm.startService()
    self.pm.addProcess('foo', ['foo'])
    self.assertIn('foo', self.pm.protocols)
    timeToDie = self.pm.protocols['foo'].transport._terminationDelay = 1
    self.reactor.advance(self.pm.threshold)
    self.pm.stopProcess('foo')
    self.reactor.advance(timeToDie)
    self.reactor.advance(0)
    self.assertEqual(self.reactor.seconds(), self.pm.timeStarted['foo'])