import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_stopProcessAlreadyStopped(self):
    """
        L{ProcessMonitor.stopProcess} silently returns if the named process
        is already stopped. eg Process has crashed and a restart has been
        rescheduled, but in the meantime, the service is stopped.
        """
    self.pm.addProcess('foo', ['foo'])
    self.assertIsNone(self.pm.stopProcess('foo'))