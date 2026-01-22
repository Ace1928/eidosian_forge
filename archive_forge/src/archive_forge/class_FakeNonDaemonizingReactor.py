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
class FakeNonDaemonizingReactor:
    """
    A dummy reactor, providing C{beforeDaemonize} and C{afterDaemonize}
    methods, but not announcing this, and logging whether the methods have been
    called.

    @ivar _beforeDaemonizeCalled: if C{beforeDaemonize} has been called or not.
    @type _beforeDaemonizeCalled: C{bool}
    @ivar _afterDaemonizeCalled: if C{afterDaemonize} has been called or not.
    @type _afterDaemonizeCalled: C{bool}
    """

    def __init__(self):
        self._beforeDaemonizeCalled = False
        self._afterDaemonizeCalled = False

    def beforeDaemonize(self):
        self._beforeDaemonizeCalled = True

    def afterDaemonize(self):
        self._afterDaemonizeCalled = True

    def addSystemEventTrigger(self, *args, **kw):
        """
        Skip event registration.
        """