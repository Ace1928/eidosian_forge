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
@skipIf(platform.isWindows(), 'Test only applies to POSIX platforms.')
def test_openFileDescriptors(self):
    """
        Processes spawned with spawnProcess() close all extraneous file
        descriptors in the parent.  They do have a stdin, stdout, and stderr
        open.
        """
    source = networkString('\nimport sys\nfrom twisted.internet import process\nsys.stdout.write(repr(process._listOpenFDs()))\nsys.stdout.flush()')
    r, w = os.pipe()
    self.addCleanup(os.close, r)
    self.addCleanup(os.close, w)
    fudgeFactor = 17
    hardResourceLimit = _getRealMaxOpenFiles()
    unlikelyFD = hardResourceLimit - fudgeFactor
    os.dup2(w, unlikelyFD)
    self.addCleanup(os.close, unlikelyFD)
    output = io.BytesIO()

    class GatheringProtocol(ProcessProtocol):
        outReceived = output.write

        def processEnded(self, reason):
            reactor.stop()
    reactor = self.buildReactor()
    reactor.callWhenRunning(reactor.spawnProcess, GatheringProtocol(), pyExe, [pyExe, b'-Wignore', b'-c', source], env=properEnv, usePTY=self.usePTY)
    self.runReactor(reactor)
    reportedChildFDs = set(eval(output.getvalue()))
    stdFDs = [0, 1, 2]
    self.assertEqual(reportedChildFDs.intersection(set(stdFDs + [unlikelyFD])), set(stdFDs))