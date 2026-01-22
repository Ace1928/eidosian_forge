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
class PosixProcessTests(unittest.TestCase, PosixProcessBase):

    def test_stderr(self):
        """
        Bytes written to stderr by the spawned process are passed to the
        C{errReceived} callback on the C{ProcessProtocol} passed to
        C{spawnProcess}.
        """
        value = '42'
        p = Accumulator()
        d = p.endedDeferred = defer.Deferred()
        reactor.spawnProcess(p, pyExe, [pyExe, b'-c', networkString("import sys; sys.stderr.write('{}')".format(value))], env=None, path='/tmp', usePTY=self.usePTY)

        def processEnded(ign):
            self.assertEqual(b'42', p.errF.getvalue())
        return d.addCallback(processEnded)

    def test_process(self):
        cmd = self.getCommand('gzip')
        s = b"there's no place like home!\n" * 3
        p = Accumulator()
        d = p.endedDeferred = defer.Deferred()
        reactor.spawnProcess(p, cmd, [cmd, b'-c'], env=None, path='/tmp', usePTY=self.usePTY)
        p.transport.write(s)
        p.transport.closeStdin()

        def processEnded(ign):
            f = p.outF
            f.seek(0, 0)
            with gzip.GzipFile(fileobj=f) as gf:
                self.assertEqual(gf.read(), s)
        return d.addCallback(processEnded)