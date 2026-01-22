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
def _applicationStartsWithConfiguredID(self, argv, uid, gid):
    """
        Assert that given a particular command line, an application is started
        as a particular UID/GID.

        @param argv: A list of strings giving the options to parse.
        @param uid: An integer giving the expected UID.
        @param gid: An integer giving the expected GID.
        """
    self.config.parseOptions(argv)
    events = []

    class FakeUnixApplicationRunner(twistd._SomeApplicationRunner):

        def setupEnvironment(self, chroot, rundir, nodaemon, umask, pidfile):
            events.append('environment')

        def shedPrivileges(self, euid, uid, gid):
            events.append(('privileges', euid, uid, gid))

        def startReactor(self, reactor, oldstdout, oldstderr):
            events.append('reactor')

        def removePID(self, pidfile):
            pass

    @implementer(service.IService, service.IProcess)
    class FakeService:
        parent = None
        running = None
        name = None
        processName = None
        uid = None
        gid = None

        def setName(self, name):
            pass

        def setServiceParent(self, parent):
            pass

        def disownServiceParent(self):
            pass

        def privilegedStartService(self):
            events.append('privilegedStartService')

        def startService(self):
            events.append('startService')

        def stopService(self):
            pass
    application = FakeService()
    verifyObject(service.IService, application)
    verifyObject(service.IProcess, application)
    runner = FakeUnixApplicationRunner(self.config)
    runner.preApplication()
    runner.application = application
    runner.postApplication()
    self.assertEqual(events, ['environment', 'privilegedStartService', ('privileges', False, uid, gid), 'startService', 'reactor'])