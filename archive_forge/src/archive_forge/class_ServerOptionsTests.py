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
class ServerOptionsTests(TestCase):
    """
    Non-platform-specific tests for the platform-specific ServerOptions class.
    """

    def test_subCommands(self):
        """
        subCommands is built from IServiceMaker plugins, and is sorted
        alphabetically.
        """

        class FakePlugin:

            def __init__(self, name):
                self.tapname = name
                self._options = 'options for ' + name
                self.description = 'description of ' + name

            def options(self):
                return self._options
        apple = FakePlugin('apple')
        banana = FakePlugin('banana')
        coconut = FakePlugin('coconut')
        donut = FakePlugin('donut')

        def getPlugins(interface):
            self.assertEqual(interface, IServiceMaker)
            yield coconut
            yield banana
            yield donut
            yield apple
        config = twistd.ServerOptions()
        self.assertEqual(config._getPlugins, plugin.getPlugins)
        config._getPlugins = getPlugins
        subCommands = config.subCommands
        expectedOrder = [apple, banana, coconut, donut]
        for subCommand, expectedCommand in zip(subCommands, expectedOrder):
            name, shortcut, parserClass, documentation = subCommand
            self.assertEqual(name, expectedCommand.tapname)
            self.assertIsNone(shortcut)
            (self.assertEqual(parserClass(), expectedCommand._options),)
            self.assertEqual(documentation, expectedCommand.description)

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

    def test_postOptionsSubCommandCausesNoSave(self):
        """
        postOptions should set no_save to True when a subcommand is used.
        """
        config = twistd.ServerOptions()
        config.subCommand = 'ueoa'
        config.postOptions()
        self.assertTrue(config['no_save'])

    def test_postOptionsNoSubCommandSavesAsUsual(self):
        """
        If no sub command is used, postOptions should not touch no_save.
        """
        config = twistd.ServerOptions()
        config.postOptions()
        self.assertFalse(config['no_save'])

    def test_listAllProfilers(self):
        """
        All the profilers that can be used in L{app.AppProfiler} are listed in
        the help output.
        """
        config = twistd.ServerOptions()
        helpOutput = str(config)
        for profiler in app.AppProfiler.profilers:
            self.assertIn(profiler, helpOutput)

    @skipIf(not _twistd_unix, 'twistd unix not available')
    def test_defaultUmask(self):
        """
        The default value for the C{umask} option is L{None}.
        """
        config = twistd.ServerOptions()
        self.assertIsNone(config['umask'])

    @skipIf(not _twistd_unix, 'twistd unix not available')
    def test_umask(self):
        """
        The value given for the C{umask} option is parsed as an octal integer
        literal.
        """
        config = twistd.ServerOptions()
        config.parseOptions(['--umask', '123'])
        self.assertEqual(config['umask'], 83)
        config.parseOptions(['--umask', '0123'])
        self.assertEqual(config['umask'], 83)

    @skipIf(not _twistd_unix, 'twistd unix not available')
    def test_invalidUmask(self):
        """
        If a value is given for the C{umask} option which cannot be parsed as
        an integer, L{UsageError} is raised by L{ServerOptions.parseOptions}.
        """
        config = twistd.ServerOptions()
        self.assertRaises(UsageError, config.parseOptions, ['--umask', 'abcdef'])

    def test_unimportableConfiguredLogObserver(self):
        """
        C{--logger} with an unimportable module raises a L{UsageError}.
        """
        config = twistd.ServerOptions()
        e = self.assertRaises(UsageError, config.parseOptions, ['--logger', 'no.such.module.I.hope'])
        self.assertTrue(e.args[0].startswith("Logger 'no.such.module.I.hope' could not be imported: 'no.such.module.I.hope' does not name an object"))
        self.assertNotIn('\n', e.args[0])

    def test_badAttributeWithConfiguredLogObserver(self):
        """
        C{--logger} with a non-existent object raises a L{UsageError}.
        """
        config = twistd.ServerOptions()
        e = self.assertRaises(UsageError, config.parseOptions, ['--logger', 'twisted.test.test_twistd.FOOBAR'])
        self.assertTrue(e.args[0].startswith("Logger 'twisted.test.test_twistd.FOOBAR' could not be imported: module 'twisted.test.test_twistd' has no attribute 'FOOBAR'"))
        self.assertNotIn('\n', e.args[0])

    def test_version(self):
        """
        C{--version} prints the version.
        """
        from twisted import copyright
        if platformType == 'win32':
            name = '(the Twisted Windows runner)'
        else:
            name = '(the Twisted daemon)'
        expectedOutput = 'twistd {} {}\n{}\n'.format(name, copyright.version, copyright.copyright)
        stdout = StringIO()
        config = twistd.ServerOptions(stdout=stdout)
        e = self.assertRaises(SystemExit, config.parseOptions, ['--version'])
        self.assertIs(e.code, None)
        self.assertEqual(stdout.getvalue(), expectedOutput)

    def test_printSubCommandForUsageError(self):
        """
        Command is printed when an invalid option is requested.
        """
        stdout = StringIO()
        config = twistd.ServerOptions(stdout=stdout)
        self.assertRaises(UsageError, config.parseOptions, ['web --foo'])