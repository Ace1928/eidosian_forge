import os
import signal
import stat
import sys
import warnings
from unittest import skipIf
from twisted.internet import error, interfaces, reactor, utils
from twisted.internet.defer import Deferred
from twisted.python.runtime import platform
from twisted.python.test.test_util import SuppressedWarningsTests
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_outputWithErrorIgnored(self):
    """
        The L{Deferred} returned by L{getProcessOutput} is fired with an
        L{IOError} L{Failure} if the child process writes to stderr.
        """
    scriptFile = self.makeSourceFile(['import sys', 'sys.stderr.write("hello world\\n")'])
    d = utils.getProcessOutput(self.exe, ['-u', scriptFile])
    d = self.assertFailure(d, IOError)

    def cbFailed(err):
        return self.assertFailure(err.processEnded, error.ProcessDone)
    d.addCallback(cbFailed)
    return d