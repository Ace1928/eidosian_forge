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
def test_timelyProcessExited(self):
    """
        If a spawned process exits, C{processExited} will be called in a
        timely manner.
        """
    reactor = self.buildReactor()

    class ExitingProtocol(ProcessProtocol):
        exited = False

        def processExited(protoSelf, reason):
            protoSelf.exited = True
            reactor.stop()
            self.assertEqual(reason.value.exitCode, 0)
    protocol = ExitingProtocol()
    reactor.callWhenRunning(reactor.spawnProcess, protocol, pyExe, [pyExe, b'-c', b'raise SystemExit(0)'], usePTY=self.usePTY)
    self.runReactor(reactor, timeout=30)
    self.assertTrue(protocol.exited)