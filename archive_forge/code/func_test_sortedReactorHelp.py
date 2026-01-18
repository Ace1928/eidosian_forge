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
def test_sortedReactorHelp(self):
    """
        Reactor names are listed alphabetically by I{--help-reactors}.
        """

    class FakeReactorInstaller:

        def __init__(self, name):
            self.shortName = 'name of ' + name
            self.description = 'description of ' + name
            self.moduleName = 'twisted.internet.default'
    apple = FakeReactorInstaller('apple')
    banana = FakeReactorInstaller('banana')
    coconut = FakeReactorInstaller('coconut')
    donut = FakeReactorInstaller('donut')

    def getReactorTypes():
        yield coconut
        yield banana
        yield donut
        yield apple
    config = twistd.ServerOptions()
    self.assertEqual(config._getReactorTypes, reactors.getReactorTypes)
    config._getReactorTypes = getReactorTypes
    config.messageOutput = StringIO()
    self.assertRaises(SystemExit, config.parseOptions, ['--help-reactors'])
    helpOutput = config.messageOutput.getvalue()
    indexes = []
    for reactor in (apple, banana, coconut, donut):

        def getIndex(s):
            self.assertIn(s, helpOutput)
            indexes.append(helpOutput.index(s))
        getIndex(reactor.shortName)
        getIndex(reactor.description)
    self.assertEqual(indexes, sorted(indexes), 'reactor descriptions were not in alphabetical order: {!r}'.format(helpOutput))