import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_startProcessAlreadyStarted(self):
    """
        L{ProcessMonitor.startProcess} silently returns if the named process is
        already started.
        """
    self.pm.addProcess('foo', ['foo'])
    self.pm.startProcess('foo')
    self.assertIsNone(self.pm.startProcess('foo'))