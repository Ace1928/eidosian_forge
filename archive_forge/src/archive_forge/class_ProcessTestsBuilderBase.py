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
class ProcessTestsBuilderBase(ReactorBuilder):
    """
    Base class for L{IReactorProcess} tests which defines some tests which
    can be applied to PTY or non-PTY uses of C{spawnProcess}.

    Subclasses are expected to set the C{usePTY} attribute to C{True} or
    C{False}.
    """
    requiredInterfaces = [IReactorProcess]
    usePTY: bool

    def test_processTransportInterface(self):
        """
        L{IReactorProcess.spawnProcess} connects the protocol passed to it
        to a transport which provides L{IProcessTransport}.
        """
        ended = Deferred()
        protocol = _ShutdownCallbackProcessProtocol(ended)
        reactor = self.buildReactor()
        transport = reactor.spawnProcess(protocol, pyExe, [pyExe, b'-c', b''], usePTY=self.usePTY)
        self.assertTrue(IProcessTransport.providedBy(transport))
        ended.addCallback(lambda ignored: reactor.stop())
        self.runReactor(reactor)

    def _writeTest(self, write):
        """
        Helper for testing L{IProcessTransport} write functionality.  This
        method spawns a child process and gives C{write} a chance to write some
        bytes to it.  It then verifies that the bytes were actually written to
        it (by relying on the child process to echo them back).

        @param write: A two-argument callable.  This is invoked with a process
            transport and some bytes to write to it.
        """
        reactor = self.buildReactor()
        ended = Deferred()
        protocol = _ShutdownCallbackProcessProtocol(ended)
        bytesToSend = b'hello, world' + networkString(os.linesep)
        program = b'import sys\nsys.stdout.write(sys.stdin.readline())\n'

        def startup():
            transport = reactor.spawnProcess(protocol, pyExe, [pyExe, b'-c', program])
            try:
                write(transport, bytesToSend)
            except BaseException:
                err(None, 'Unhandled exception while writing')
                transport.signalProcess('KILL')
        reactor.callWhenRunning(startup)
        ended.addCallback(lambda ignored: reactor.stop())
        self.runReactor(reactor)
        self.assertEqual(bytesToSend, b''.join(protocol.received[1]))

    def test_write(self):
        """
        L{IProcessTransport.write} writes the specified C{bytes} to the standard
        input of the child process.
        """

        def write(transport, bytesToSend):
            transport.write(bytesToSend)
        self._writeTest(write)

    def test_writeSequence(self):
        """
        L{IProcessTransport.writeSequence} writes the specified C{list} of
        C{bytes} to the standard input of the child process.
        """

        def write(transport, bytesToSend):
            transport.writeSequence([bytesToSend])
        self._writeTest(write)

    def test_writeToChild(self):
        """
        L{IProcessTransport.writeToChild} writes the specified C{bytes} to the
        specified file descriptor of the child process.
        """

        def write(transport, bytesToSend):
            transport.writeToChild(0, bytesToSend)
        self._writeTest(write)

    def test_writeToChildBadFileDescriptor(self):
        """
        L{IProcessTransport.writeToChild} raises L{KeyError} if passed a file
        descriptor which is was not set up by L{IReactorProcess.spawnProcess}.
        """

        def write(transport, bytesToSend):
            try:
                self.assertRaises(KeyError, transport.writeToChild, 13, bytesToSend)
            finally:
                transport.write(bytesToSend)
        self._writeTest(write)

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

    @onlyOnPOSIX
    def test_errorDuringExec(self):
        """
        When L{os.execvpe} raises an exception, it will format that exception
        on stderr as UTF-8, regardless of system encoding information.
        """

        def execvpe(*args, **kw):
            filename = '<☃>'
            if not isinstance(filename, str):
                filename = filename.encode('utf-8')
            codeobj = compile('1/0', filename, 'single')
            eval(codeobj)
        self.patch(os, 'execvpe', execvpe)
        self.patch(sys, 'getfilesystemencoding', lambda: 'ascii')
        reactor = self.buildReactor()
        reactor._neverUseSpawn = True
        output = io.BytesIO()
        expectedFD = 1 if self.usePTY else 2

        @reactor.callWhenRunning
        def whenRunning():

            class TracebackCatcher(ProcessProtocol):

                def childDataReceived(self, child, data):
                    if child == expectedFD:
                        output.write(data)

                def processEnded(self, reason):
                    reactor.stop()
            reactor.spawnProcess(TracebackCatcher(), pyExe, [pyExe, b'-c', b''], usePTY=self.usePTY)
        self.runReactor(reactor, timeout=30)
        self.assertIn('☃'.encode(), output.getvalue())

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

    @skipIf(_uidgidSkip, _uidgidSkipReason)
    def test_changeUID(self):
        """
        If a value is passed for L{IReactorProcess.spawnProcess}'s C{uid}, the
        child process is run with that UID.
        """
        self._changeIDTest('uid')

    @skipIf(_uidgidSkip, _uidgidSkipReason)
    def test_changeGID(self):
        """
        If a value is passed for L{IReactorProcess.spawnProcess}'s C{gid}, the
        child process is run with that GID.
        """
        self._changeIDTest('gid')

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