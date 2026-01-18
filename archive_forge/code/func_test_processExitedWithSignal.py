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
def test_processExitedWithSignal(self):
    """
        The C{reason} argument passed to L{IProcessProtocol.processExited} is a
        L{ProcessTerminated} instance if the child process exits with a signal.
        """
    sigName = 'TERM'
    sigNum = getattr(signal, 'SIG' + sigName)
    exited = Deferred()
    source = b"import sys\nsys.stdout.write('x')\nsys.stdout.flush()\nsys.stdin.read()\n"

    class Exiter(ProcessProtocol):

        def childDataReceived(self, fd, data):
            msg('childDataReceived(%d, %r)' % (fd, data))
            self.transport.signalProcess(sigName)

        def childConnectionLost(self, fd):
            msg('childConnectionLost(%d)' % (fd,))

        def processExited(self, reason):
            msg(f'processExited({reason!r})')
            exited.callback([reason])

        def processEnded(self, reason):
            msg(f'processEnded({reason!r})')
    reactor = self.buildReactor()
    reactor.callWhenRunning(reactor.spawnProcess, Exiter(), pyExe, [pyExe, b'-c', source], usePTY=self.usePTY)

    def cbExited(args):
        failure, = args
        failure.trap(ProcessTerminated)
        err = failure.value
        if platform.isWindows():
            self.assertIsNone(err.signal)
            self.assertEqual(err.exitCode, 1)
        else:
            self.assertEqual(err.signal, sigNum)
            self.assertIsNone(err.exitCode)
    exited.addCallback(cbExited)
    exited.addErrback(err)
    exited.addCallback(lambda ign: reactor.stop())
    self.runReactor(reactor)