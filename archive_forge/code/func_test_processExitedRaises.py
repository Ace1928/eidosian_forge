import io
import os
import signal
import subprocess
import sys
import threading
from unittest import skipIf
import hamcrest
from twisted.internet import utils
from twisted.internet.defer import Deferred, inlineCallbacks, succeed
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.internet.interfaces import IProcessTransport, IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath, _asFilesystemBytes
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.test.test_process import Accumulator
from twisted.trial.unittest import SynchronousTestCase, TestCase
import sys
from twisted.internet import process
def test_processExitedRaises(self):
    """
        If L{IProcessProtocol.processExited} raises an exception, it is logged.
        """
    reactor = self.buildReactor()

    class TestException(Exception):
        pass

    class Protocol(ProcessProtocol):

        def processExited(self, reason):
            reactor.stop()
            raise TestException('processedExited raised')
    protocol = Protocol()
    transport = reactor.spawnProcess(protocol, pyExe, [pyExe, b'-c', b''], usePTY=self.usePTY)
    self.runReactor(reactor)
    if process is not None:
        for pid, handler in list(process.reapProcessHandlers.items()):
            if handler is not transport:
                continue
            process.unregisterReapProcessHandler(pid, handler)
            self.fail('After processExited raised, transport was left in reapProcessHandlers')
    self.assertEqual(1, len(self.flushLoggedErrors(TestException)))