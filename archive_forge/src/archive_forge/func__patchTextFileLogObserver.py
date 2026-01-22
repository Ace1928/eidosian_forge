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
def _patchTextFileLogObserver(patch):
    """
    Patch L{logger.textFileLogObserver} to record every call and keep a
    reference to the passed log file for tests.

    @param patch: a callback for patching (usually L{TestCase.patch}).

    @return: the list that keeps track of the log files.
    @rtype: C{list}
    """
    logFiles = []
    oldFileLogObserver = logger.textFileLogObserver

    def observer(logFile, *args, **kwargs):
        logFiles.append(logFile)
        return oldFileLogObserver(logFile, *args, **kwargs)
    patch(logger, 'textFileLogObserver', observer)
    return logFiles