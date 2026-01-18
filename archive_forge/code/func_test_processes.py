import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_processes(self):
    """
        Accessing L{ProcessMonitor.processes} results in deprecation warning

        Even when there are no processes, and thus no process is converted
        to a tuple, accessing the L{ProcessMonitor.processes} property
        should generate its own DeprecationWarning.
        """
    myProcesses = self.pm.processes
    self.assertEquals(myProcesses, {})
    warnings = self.flushWarnings()
    first = warnings.pop(0)
    self.assertIs(first['category'], DeprecationWarning)
    self.assertEquals(warnings, [])