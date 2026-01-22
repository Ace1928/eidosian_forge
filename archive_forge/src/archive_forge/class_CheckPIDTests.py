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
class CheckPIDTests(TestCase):
    """
    Tests for L{checkPID}.
    """

    def test_notExists(self):
        """
        Nonexistent PID file is not an error.
        """
        self.patch(os.path, 'exists', lambda _: False)
        checkPID('non-existent PID file')

    def test_nonNumeric(self):
        """
        Non-numeric content in a PID file causes a system exit.
        """
        pidfile = self.mktemp()
        with open(pidfile, 'w') as f:
            f.write('non-numeric')
        e = self.assertRaises(SystemExit, checkPID, pidfile)
        self.assertIn('non-numeric value', e.code)

    def test_anotherRunning(self):
        """
        Another running twistd server causes a system exit.
        """
        pidfile = self.mktemp()
        with open(pidfile, 'w') as f:
            f.write('42')

        def kill(pid, sig):
            pass
        self.patch(os, 'kill', kill)
        e = self.assertRaises(SystemExit, checkPID, pidfile)
        self.assertIn('Another twistd server', e.code)

    def test_stale(self):
        """
        Stale PID file is removed without causing a system exit.
        """
        pidfile = self.mktemp()
        with open(pidfile, 'w') as f:
            f.write(str(os.getpid() + 1))

        def kill(pid, sig):
            raise OSError(errno.ESRCH, 'fake')
        self.patch(os, 'kill', kill)
        checkPID(pidfile)
        self.assertFalse(os.path.exists(pidfile))

    def test_unexpectedOSError(self):
        """
        An unexpected L{OSError} when checking the validity of a
        PID in a C{pidfile} terminates the process via L{SystemExit}.
        """
        pidfile = self.mktemp()
        with open(pidfile, 'w') as f:
            f.write('3581')

        def kill(pid, sig):
            raise OSError(errno.EBADF, 'fake')
        self.patch(os, 'kill', kill)
        e = self.assertRaises(SystemExit, checkPID, pidfile)
        self.assertIsNot(e.code, None)
        self.assertTrue(e.args[0].startswith("Can't check status of PID"))