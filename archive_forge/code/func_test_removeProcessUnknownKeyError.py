import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_removeProcessUnknownKeyError(self):
    """
        L{ProcessMonitor.removeProcess} raises a C{KeyError} if the given
        process name isn't recognised.
        """
    self.pm.startService()
    self.assertRaises(KeyError, self.pm.removeProcess, 'foo')