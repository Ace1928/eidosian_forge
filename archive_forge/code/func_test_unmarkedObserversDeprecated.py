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
def test_unmarkedObserversDeprecated(self):
    """
        L{app.AppLogger} using a logger observer which does not implement
        L{ILogObserver} or L{LegacyILogObserver} will be wrapped in a compat
        shim and raise a L{DeprecationWarning}.
        """
    logs = []
    logger = app.AppLogger({})
    logger._getLogObserver = lambda: logs.append
    logger.start(Componentized())
    self.assertIn('starting up', textFromEventDict(logs[0]))
    warnings = self.flushWarnings([self.test_unmarkedObserversDeprecated])
    self.assertEqual(warnings[0]['message'], 'Passing a logger factory which makes log observers which do not implement twisted.logger.ILogObserver or twisted.python.log.ILogObserver to twisted.application.app.AppLogger was deprecated in Twisted 16.2. Please use a factory that produces twisted.logger.ILogObserver (or the legacy twisted.python.log.ILogObserver) implementing objects instead.')
    self.assertEqual(len(warnings), 1, warnings)