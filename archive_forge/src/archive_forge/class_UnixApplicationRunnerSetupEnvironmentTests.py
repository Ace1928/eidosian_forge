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
@skipIf(not _twistd_unix, 'twistd unix not available')
class UnixApplicationRunnerSetupEnvironmentTests(TestCase):
    """
    Tests for L{UnixApplicationRunner.setupEnvironment}.

    @ivar root: The root of the filesystem, or C{unset} if none has been
        specified with a call to L{os.chroot} (patched for this TestCase with
        L{UnixApplicationRunnerSetupEnvironmentTests.chroot}).

    @ivar cwd: The current working directory of the process, or C{unset} if
        none has been specified with a call to L{os.chdir} (patched for this
        TestCase with L{UnixApplicationRunnerSetupEnvironmentTests.chdir}).

    @ivar mask: The current file creation mask of the process, or C{unset} if
        none has been specified with a call to L{os.umask} (patched for this
        TestCase with L{UnixApplicationRunnerSetupEnvironmentTests.umask}).

    @ivar daemon: A boolean indicating whether daemonization has been performed
        by a call to L{_twistd_unix.daemonize} (patched for this TestCase with
        L{UnixApplicationRunnerSetupEnvironmentTests}.
    """
    unset = object()

    def setUp(self):
        self.root = self.unset
        self.cwd = self.unset
        self.mask = self.unset
        self.daemon = False
        self.pid = os.getpid()
        self.patch(os, 'chroot', lambda path: setattr(self, 'root', path))
        self.patch(os, 'chdir', lambda path: setattr(self, 'cwd', path))
        self.patch(os, 'umask', lambda mask: setattr(self, 'mask', mask))
        self.runner = UnixApplicationRunner(twistd.ServerOptions())
        self.runner.daemonize = self.daemonize

    def daemonize(self, reactor):
        """
        Indicate that daemonization has happened and change the PID so that the
        value written to the pidfile can be tested in the daemonization case.
        """
        self.daemon = True
        self.patch(os, 'getpid', lambda: self.pid + 1)

    def test_chroot(self):
        """
        L{UnixApplicationRunner.setupEnvironment} changes the root of the
        filesystem if passed a non-L{None} value for the C{chroot} parameter.
        """
        self.runner.setupEnvironment('/foo/bar', '.', True, None, None)
        self.assertEqual(self.root, '/foo/bar')

    def test_noChroot(self):
        """
        L{UnixApplicationRunner.setupEnvironment} does not change the root of
        the filesystem if passed L{None} for the C{chroot} parameter.
        """
        self.runner.setupEnvironment(None, '.', True, None, None)
        self.assertIs(self.root, self.unset)

    def test_changeWorkingDirectory(self):
        """
        L{UnixApplicationRunner.setupEnvironment} changes the working directory
        of the process to the path given for the C{rundir} parameter.
        """
        self.runner.setupEnvironment(None, '/foo/bar', True, None, None)
        self.assertEqual(self.cwd, '/foo/bar')

    def test_daemonize(self):
        """
        L{UnixApplicationRunner.setupEnvironment} daemonizes the process if
        C{False} is passed for the C{nodaemon} parameter.
        """
        with AlternateReactor(FakeDaemonizingReactor()):
            self.runner.setupEnvironment(None, '.', False, None, None)
        self.assertTrue(self.daemon)

    def test_noDaemonize(self):
        """
        L{UnixApplicationRunner.setupEnvironment} does not daemonize the
        process if C{True} is passed for the C{nodaemon} parameter.
        """
        self.runner.setupEnvironment(None, '.', True, None, None)
        self.assertFalse(self.daemon)

    def test_nonDaemonPIDFile(self):
        """
        L{UnixApplicationRunner.setupEnvironment} writes the process's PID to
        the file specified by the C{pidfile} parameter.
        """
        pidfile = self.mktemp()
        self.runner.setupEnvironment(None, '.', True, None, pidfile)
        with open(pidfile, 'rb') as f:
            pid = int(f.read())
        self.assertEqual(pid, self.pid)

    def test_daemonPIDFile(self):
        """
        L{UnixApplicationRunner.setupEnvironment} writes the daemonized
        process's PID to the file specified by the C{pidfile} parameter if
        C{nodaemon} is C{False}.
        """
        pidfile = self.mktemp()
        with AlternateReactor(FakeDaemonizingReactor()):
            self.runner.setupEnvironment(None, '.', False, None, pidfile)
        with open(pidfile, 'rb') as f:
            pid = int(f.read())
        self.assertEqual(pid, self.pid + 1)

    def test_umask(self):
        """
        L{UnixApplicationRunner.setupEnvironment} changes the process umask to
        the value specified by the C{umask} parameter.
        """
        with AlternateReactor(FakeDaemonizingReactor()):
            self.runner.setupEnvironment(None, '.', False, 123, None)
        self.assertEqual(self.mask, 123)

    def test_noDaemonizeNoUmask(self):
        """
        L{UnixApplicationRunner.setupEnvironment} doesn't change the process
        umask if L{None} is passed for the C{umask} parameter and C{True} is
        passed for the C{nodaemon} parameter.
        """
        self.runner.setupEnvironment(None, '.', True, None, None)
        self.assertIs(self.mask, self.unset)

    def test_daemonizedNoUmask(self):
        """
        L{UnixApplicationRunner.setupEnvironment} changes the process umask to
        C{0077} if L{None} is passed for the C{umask} parameter and C{False} is
        passed for the C{nodaemon} parameter.
        """
        with AlternateReactor(FakeDaemonizingReactor()):
            self.runner.setupEnvironment(None, '.', False, None, None)
        self.assertEqual(self.mask, 63)