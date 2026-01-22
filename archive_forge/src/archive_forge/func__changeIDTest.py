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
def _changeIDTest(self, which):
    """
        Launch a child process, using either the C{uid} or C{gid} argument to
        L{IReactorProcess.spawnProcess} to change either its UID or GID to a
        different value.  If the child process reports this hasn't happened,
        raise an exception to fail the test.

        @param which: Either C{b"uid"} or C{b"gid"}.
        """
    program = ['import os', f'raise SystemExit(os.get{which}() != 1)']
    container = []

    class CaptureExitStatus(ProcessProtocol):

        def processEnded(self, reason):
            container.append(reason)
            reactor.stop()
    reactor = self.buildReactor()
    protocol = CaptureExitStatus()
    reactor.callWhenRunning(reactor.spawnProcess, protocol, pyExe, [pyExe, '-c', '\n'.join(program)], **{which: 1})
    self.runReactor(reactor)
    self.assertEqual(0, container[0].value.exitCode)