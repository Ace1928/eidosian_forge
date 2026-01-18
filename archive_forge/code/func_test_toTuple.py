import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_toTuple(self):
    """
        _Process.toTuple is deprecated.

        When getting the deprecated processes property, the actual
        data (kept in the class _Process) is converted to a tuple --
        which produces a DeprecationWarning per process so converted.
        """
    self.pm.addProcess('foo', ['foo'])
    myprocesses = self.pm.processes
    self.assertEquals(len(myprocesses), 1)
    warnings = self.flushWarnings()
    foundToTuple = False
    for warning in warnings:
        self.assertIs(warning['category'], DeprecationWarning)
        if 'toTuple' in warning['message']:
            foundToTuple = True
    self.assertTrue(foundToTuple, f'no tuple deprecation found:{repr(warnings)}')