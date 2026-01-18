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
@skipIf(getattr(signal, 'SIGCHLD', None) is None, "Platform lacks SIGCHLD, early-spawnProcess test can't work.")
def test_spawnProcessEarlyIsReaped(self):
    """
        If, before the reactor is started with L{IReactorCore.run}, a
        process is started with L{IReactorProcess.spawnProcess} and
        terminates, the process is reaped once the reactor is started.
        """
    reactor = self.buildReactor()
    if self.usePTY:
        childFDs = None
    else:
        childFDs = {}
    signaled = threading.Event()

    def handler(*args):
        signaled.set()
    signal.signal(signal.SIGCHLD, handler)
    ended = Deferred()
    reactor.spawnProcess(_ShutdownCallbackProcessProtocol(ended), pyExe, [pyExe, b'-c', b''], usePTY=self.usePTY, childFDs=childFDs)
    signaled.wait(120)
    if not signaled.isSet():
        self.fail('Timed out waiting for child process to exit.')
    result = []
    ended.addCallback(result.append)
    if result:
        return
    ended.addCallback(lambda ignored: reactor.stop())
    self.runReactor(reactor)
    self.assertTrue(result)