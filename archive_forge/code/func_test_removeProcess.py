import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_removeProcess(self):
    """
        L{ProcessMonitor.removeProcess} removes the process from the public
        processes list.
        """
    self.pm.startService()
    self.pm.addProcess('foo', ['foo'])
    self.assertEqual(len(self.pm.processes), 1)
    self.pm.removeProcess('foo')
    self.assertEqual(len(self.pm.processes), 0)