import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
@skipIf(not interfaces.IReactorProcess(reactor, None), "reactor doesn't support IReactorProcess")
class ClosingPipesTests(unittest.TestCase):

    def doit(self, fd):
        """
        Create a child process and close one of its output descriptors using
        L{IProcessTransport.closeStdout} or L{IProcessTransport.closeStderr}.
        Return a L{Deferred} which fires after verifying that the descriptor was
        really closed.
        """
        p = ClosingPipesProcessProtocol(True)
        self.assertFailure(p.deferred, error.ProcessTerminated)
        p.deferred.addCallback(self._endProcess, p)
        reactor.spawnProcess(p, pyExe, [pyExe, b'-u', b'-c', networkString('input()\nimport sys, os, time\nfor i in range(1000):\n    os.write(%d, b"foo\\n")\n    time.sleep(0.01)\nsys.exit(42)\n' % (fd,))], env=None)
        if fd == 1:
            p.transport.closeStdout()
        elif fd == 2:
            p.transport.closeStderr()
        else:
            raise RuntimeError
        p.transport.write(b'go\n')
        p.transport.closeStdin()
        return p.deferred

    def _endProcess(self, reason, p):
        """
        Check that a failed write prevented the process from getting to its
        custom exit code.
        """
        self.assertNotEqual(reason.exitCode, 42, 'process reason was %r' % reason)
        self.assertEqual(p.output, b'')
        return p.errput

    def test_stdout(self):
        """
        ProcessProtocol.transport.closeStdout actually closes the pipe.
        """
        d = self.doit(1)

        def _check(errput):
            if runtime.platform.isWindows():
                self.assertIn(b'OSError', errput)
                self.assertIn(b'22', errput)
            else:
                self.assertIn(b'BrokenPipeError', errput)
            if runtime.platform.getType() != 'win32':
                self.assertIn(b'Broken pipe', errput)
        d.addCallback(_check)
        return d

    def test_stderr(self):
        """
        ProcessProtocol.transport.closeStderr actually closes the pipe.
        """
        d = self.doit(2)

        def _check(errput):
            self.assertEqual(errput, b'')
        d.addCallback(_check)
        return d