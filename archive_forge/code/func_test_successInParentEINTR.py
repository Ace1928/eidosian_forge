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