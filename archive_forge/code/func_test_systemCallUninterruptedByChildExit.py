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
def test_systemCallUninterruptedByChildExit(self):
    """
        If a child process exits while a system call is in progress, the system
        call should not be interfered with.  In particular, it should not fail
        with EINTR.

        Older versions of Twisted installed a SIGCHLD handler on POSIX without
        using the feature exposed by the SA_RESTART flag to sigaction(2).  The
        most noticeable problem this caused was for blocking reads and writes to
        sometimes fail with EINTR.
        """
    reactor = self.buildReactor()
    result = []

    def f():
        try:
            exe = pyExe.decode(sys.getfilesystemencoding())
            subprocess.Popen([exe, '-c', 'import time; time.sleep(0.1)'])
            f2 = subprocess.Popen([exe, '-c', "import time; time.sleep(0.5);print('Foo')"], stdout=subprocess.PIPE)
            with f2.stdout:
                result.append(f2.stdout.read())
        finally:
            reactor.stop()
    reactor.callWhenRunning(f)
    self.runReactor(reactor)
    self.assertEqual(result, [b'Foo' + os.linesep.encode('ascii')])