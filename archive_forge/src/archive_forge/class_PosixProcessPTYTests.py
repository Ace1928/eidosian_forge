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
@skipIf(runtime.platform.getType() != 'posix', 'Only runs on POSIX platform')
@skipIf(not interfaces.IReactorProcess(reactor, None), "reactor doesn't support IReactorProcess")
class PosixProcessPTYTests(unittest.TestCase, PosixProcessBase):
    """
    Just like PosixProcessTests, but use ptys instead of pipes.
    """
    usePTY = True

    def test_openingTTY(self):
        scriptPath = b'twisted.test.process_tty'
        p = Accumulator()
        d = p.endedDeferred = defer.Deferred()
        reactor.spawnProcess(p, pyExe, [pyExe, b'-u', b'-m', scriptPath], env=properEnv, usePTY=self.usePTY)
        p.transport.write(b'hello world!\n')

        def processEnded(ign):
            self.assertRaises(error.ProcessExitedAlready, p.transport.signalProcess, 'HUP')
            self.assertEqual(p.outF.getvalue(), b'hello world!\r\nhello world!\r\n', 'Error message from process_tty follows:\n\n%s\n\n' % (p.outF.getvalue(),))
        return d.addCallback(processEnded)

    def test_badArgs(self):
        pyArgs = [pyExe, b'-u', b'-c', b"print('hello')"]
        p = Accumulator()
        self.assertRaises(ValueError, reactor.spawnProcess, p, pyExe, pyArgs, usePTY=1, childFDs={1: b'r'})