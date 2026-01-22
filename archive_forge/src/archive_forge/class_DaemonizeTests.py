import errno
import inspect
import os
import pickle
import signal
import sys
from io import StringIO
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted import internet, logger, plugin
from twisted.application import app, reactors, service
from twisted.application.service import IServiceMaker
from twisted.internet.base import ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReactorDaemonize, _ISupportsExitSignalCapturing
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.internet.testing import MemoryReactor
from twisted.logger import ILogObserver, globalLogBeginner, globalLogPublisher
from twisted.python import util
from twisted.python.components import Componentized
from twisted.python.fakepwd import UserDatabase
from twisted.python.log import ILogObserver as LegacyILogObserver, textFromEventDict
from twisted.python.reflect import requireModule
from twisted.python.runtime import platformType
from twisted.python.usage import UsageError
from twisted.scripts import twistd
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
@skipIf(not _twistd_unix, 'twistd unix support not available')
class DaemonizeTests(TestCase):
    """
    Tests for L{_twistd_unix.UnixApplicationRunner} daemonization.
    """

    def setUp(self):
        self.mockos = MockOS()
        self.config = twistd.ServerOptions()
        self.patch(_twistd_unix, 'os', self.mockos)
        self.runner = _twistd_unix.UnixApplicationRunner(self.config)
        self.runner.application = service.Application('Hi!')
        self.runner.oldstdout = sys.stdout
        self.runner.oldstderr = sys.stderr
        self.runner.startReactor = lambda *args: None

    def test_success(self):
        """
        When double fork succeeded in C{daemonize}, the child process writes
        B{0} to the status pipe.
        """
        with AlternateReactor(FakeDaemonizingReactor()):
            self.runner.postApplication()
        self.assertEqual(self.mockos.actions, [('chdir', '.'), ('umask', 63), ('fork', True), 'setsid', ('fork', True), ('write', -2, b'0'), ('unlink', 'twistd.pid')])
        self.assertEqual(self.mockos.closed, [-3, -2])

    def test_successInParent(self):
        """
        The parent process initiating the C{daemonize} call reads data from the
        status pipe and then exit the process.
        """
        self.mockos.child = False
        self.mockos.readData = b'0'
        with AlternateReactor(FakeDaemonizingReactor()):
            self.assertRaises(SystemError, self.runner.postApplication)
        self.assertEqual(self.mockos.actions, [('chdir', '.'), ('umask', 63), ('fork', True), ('read', -1, 100), ('exit', 0), ('unlink', 'twistd.pid')])
        self.assertEqual(self.mockos.closed, [-1])

    def test_successEINTR(self):
        """
        If the C{os.write} call to the status pipe raises an B{EINTR} error,
        the process child retries to write.
        """
        written = []

        def raisingWrite(fd, data):
            written.append((fd, data))
            if len(written) == 1:
                raise OSError(errno.EINTR)
        self.mockos.write = raisingWrite
        with AlternateReactor(FakeDaemonizingReactor()):
            self.runner.postApplication()
        self.assertEqual(self.mockos.actions, [('chdir', '.'), ('umask', 63), ('fork', True), 'setsid', ('fork', True), ('unlink', 'twistd.pid')])
        self.assertEqual(self.mockos.closed, [-3, -2])
        self.assertEqual([(-2, b'0'), (-2, b'0')], written)

    def test_successInParentEINTR(self):
        """
        If the C{os.read} call on the status pipe raises an B{EINTR} error, the
        parent child retries to read.
        """
        read = []

        def raisingRead(fd, size):
            read.append((fd, size))
            if len(read) == 1:
                raise OSError(errno.EINTR)
            return b'0'
        self.mockos.read = raisingRead
        self.mockos.child = False
        with AlternateReactor(FakeDaemonizingReactor()):
            self.assertRaises(SystemError, self.runner.postApplication)
        self.assertEqual(self.mockos.actions, [('chdir', '.'), ('umask', 63), ('fork', True), ('exit', 0), ('unlink', 'twistd.pid')])
        self.assertEqual(self.mockos.closed, [-1])
        self.assertEqual([(-1, 100), (-1, 100)], read)

    def assertErrorWritten(self, raised, reported):
        """
        Assert L{UnixApplicationRunner.postApplication} writes
        C{reported} to its status pipe if the service raises an
        exception whose message is C{raised}.
        """

        class FakeService(service.Service):

            def startService(self):
                raise RuntimeError(raised)
        errorService = FakeService()
        errorService.setServiceParent(self.runner.application)
        with AlternateReactor(FakeDaemonizingReactor()):
            self.assertRaises(RuntimeError, self.runner.postApplication)
        self.assertEqual(self.mockos.actions, [('chdir', '.'), ('umask', 63), ('fork', True), 'setsid', ('fork', True), ('write', -2, reported), ('unlink', 'twistd.pid')])
        self.assertEqual(self.mockos.closed, [-3, -2])

    def test_error(self):
        """
        If an error happens during daemonization, the child process writes the
        exception error to the status pipe.
        """
        self.assertErrorWritten(raised='Something is wrong', reported=b'1 RuntimeError: Something is wrong')

    def test_unicodeError(self):
        """
        If an error happens during daemonization, and that error's
        message is Unicode, the child encodes the message as ascii
        with backslash Unicode code points.
        """
        self.assertErrorWritten(raised='•', reported=b'1 RuntimeError: \\u2022')

    def assertErrorInParentBehavior(self, readData, errorMessage, mockOSActions):
        """
        Make L{os.read} appear to return C{readData}, and assert that
        L{UnixApplicationRunner.postApplication} writes
        C{errorMessage} to standard error and executes the calls
        against L{os} functions specified in C{mockOSActions}.
        """
        self.mockos.child = False
        self.mockos.readData = readData
        errorIO = StringIO()
        self.patch(sys, '__stderr__', errorIO)
        with AlternateReactor(FakeDaemonizingReactor()):
            self.assertRaises(SystemError, self.runner.postApplication)
        self.assertEqual(errorIO.getvalue(), errorMessage)
        self.assertEqual(self.mockos.actions, mockOSActions)
        self.assertEqual(self.mockos.closed, [-1])

    def test_errorInParent(self):
        """
        When the child writes an error message to the status pipe
        during daemonization, the parent writes the repr of the
        message to C{stderr} and exits with non-zero status code.
        """
        self.assertErrorInParentBehavior(readData=b'1 Exception: An identified error', errorMessage="An error has occurred: b'Exception: An identified error'\nPlease look at log file for more information.\n", mockOSActions=[('chdir', '.'), ('umask', 63), ('fork', True), ('read', -1, 100), ('exit', 1), ('unlink', 'twistd.pid')])

    def test_nonASCIIErrorInParent(self):
        """
        When the child writes a non-ASCII error message to the status
        pipe during daemonization, the parent writes the repr of the
        message to C{stderr} and exits with a non-zero status code.
        """
        self.assertErrorInParentBehavior(readData=b'1 Exception: \xff', errorMessage="An error has occurred: b'Exception: \\xff'\nPlease look at log file for more information.\n", mockOSActions=[('chdir', '.'), ('umask', 63), ('fork', True), ('read', -1, 100), ('exit', 1), ('unlink', 'twistd.pid')])

    def test_errorInParentWithTruncatedUnicode(self):
        """
        When the child writes a non-ASCII error message to the status
        pipe during daemonization, and that message is too longer, the
        parent writes the repr of the truncated message to C{stderr}
        and exits with a non-zero status code.
        """
        truncatedMessage = b'1 RuntimeError: ' + b'\\u2022' * 14
        reportedMessage = "b'RuntimeError: {}'".format('\\\\u2022' * 14)
        self.assertErrorInParentBehavior(readData=truncatedMessage, errorMessage='An error has occurred: {}\nPlease look at log file for more information.\n'.format(reportedMessage), mockOSActions=[('chdir', '.'), ('umask', 63), ('fork', True), ('read', -1, 100), ('exit', 1), ('unlink', 'twistd.pid')])

    def test_errorMessageTruncated(self):
        """
        If an error occurs during daemonization and its message is too
        long, it's truncated by the child.
        """
        self.assertErrorWritten(raised='x' * 200, reported=b'1 RuntimeError: ' + b'x' * 84)

    def test_unicodeErrorMessageTruncated(self):
        """
        If an error occurs during daemonization and its message is
        unicode and too long, it's truncated by the child, even if
        this splits a unicode escape sequence.
        """
        self.assertErrorWritten(raised='•' * 30, reported=b'1 RuntimeError: ' + b'\\u2022' * 14)

    def test_hooksCalled(self):
        """
        C{daemonize} indeed calls L{IReactorDaemonize.beforeDaemonize} and
        L{IReactorDaemonize.afterDaemonize} if the reactor implements
        L{IReactorDaemonize}.
        """
        reactor = FakeDaemonizingReactor()
        self.runner.daemonize(reactor)
        self.assertTrue(reactor._beforeDaemonizeCalled)
        self.assertTrue(reactor._afterDaemonizeCalled)

    def test_hooksNotCalled(self):
        """
        C{daemonize} does NOT call L{IReactorDaemonize.beforeDaemonize} or
        L{IReactorDaemonize.afterDaemonize} if the reactor does NOT implement
        L{IReactorDaemonize}.
        """
        reactor = FakeNonDaemonizingReactor()
        self.runner.daemonize(reactor)
        self.assertFalse(reactor._beforeDaemonizeCalled)
        self.assertFalse(reactor._afterDaemonizeCalled)