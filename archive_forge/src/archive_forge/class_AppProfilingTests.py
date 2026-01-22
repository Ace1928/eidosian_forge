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
class AppProfilingTests(TestCase):
    """
    Tests for L{app.AppProfiler}.
    """

    @skipIf(not profile, 'profile module not available')
    def test_profile(self):
        """
        L{app.ProfileRunner.run} should call the C{run} method of the reactor
        and save profile data in the specified file.
        """
        config = twistd.ServerOptions()
        config['profile'] = self.mktemp()
        config['profiler'] = 'profile'
        profiler = app.AppProfiler(config)
        reactor = DummyReactor()
        profiler.run(reactor)
        self.assertTrue(reactor.called)
        with open(config['profile']) as f:
            data = f.read()
        self.assertIn('DummyReactor.run', data)
        self.assertIn('function calls', data)

    def _testStats(self, statsClass, profile):
        out = StringIO()
        stdout = self.patch(sys, 'stdout', out)
        stats = statsClass(profile)
        stats.print_stats()
        stdout.restore()
        data = out.getvalue()
        self.assertIn('function calls', data)
        self.assertIn('(run)', data)

    @skipIf(not profile, 'profile module not available')
    def test_profileSaveStats(self):
        """
        With the C{savestats} option specified, L{app.ProfileRunner.run}
        should save the raw stats object instead of a summary output.
        """
        config = twistd.ServerOptions()
        config['profile'] = self.mktemp()
        config['profiler'] = 'profile'
        config['savestats'] = True
        profiler = app.AppProfiler(config)
        reactor = DummyReactor()
        profiler.run(reactor)
        self.assertTrue(reactor.called)
        self._testStats(pstats.Stats, config['profile'])

    def test_withoutProfile(self):
        """
        When the C{profile} module is not present, L{app.ProfilerRunner.run}
        should raise a C{SystemExit} exception.
        """
        savedModules = sys.modules.copy()
        config = twistd.ServerOptions()
        config['profiler'] = 'profile'
        profiler = app.AppProfiler(config)
        sys.modules['profile'] = None
        try:
            self.assertRaises(SystemExit, profiler.run, None)
        finally:
            sys.modules.clear()
            sys.modules.update(savedModules)

    @skipIf(not profile, 'profile module not available')
    def test_profilePrintStatsError(self):
        """
        When an error happens during the print of the stats, C{sys.stdout}
        should be restored to its initial value.
        """

        class ErroneousProfile(profile.Profile):

            def print_stats(self):
                raise RuntimeError('Boom')
        self.patch(profile, 'Profile', ErroneousProfile)
        config = twistd.ServerOptions()
        config['profile'] = self.mktemp()
        config['profiler'] = 'profile'
        profiler = app.AppProfiler(config)
        reactor = DummyReactor()
        oldStdout = sys.stdout
        self.assertRaises(RuntimeError, profiler.run, reactor)
        self.assertIs(sys.stdout, oldStdout)

    @skipIf(not cProfile, 'cProfile module not available')
    def test_cProfile(self):
        """
        L{app.CProfileRunner.run} should call the C{run} method of the
        reactor and save profile data in the specified file.
        """
        config = twistd.ServerOptions()
        config['profile'] = self.mktemp()
        config['profiler'] = 'cProfile'
        profiler = app.AppProfiler(config)
        reactor = DummyReactor()
        profiler.run(reactor)
        self.assertTrue(reactor.called)
        with open(config['profile']) as f:
            data = f.read()
        self.assertIn('run', data)
        self.assertIn('function calls', data)

    @skipIf(not cProfile, 'cProfile module not available')
    def test_cProfileSaveStats(self):
        """
        With the C{savestats} option specified,
        L{app.CProfileRunner.run} should save the raw stats object
        instead of a summary output.
        """
        config = twistd.ServerOptions()
        config['profile'] = self.mktemp()
        config['profiler'] = 'cProfile'
        config['savestats'] = True
        profiler = app.AppProfiler(config)
        reactor = DummyReactor()
        profiler.run(reactor)
        self.assertTrue(reactor.called)
        self._testStats(pstats.Stats, config['profile'])

    def test_withoutCProfile(self):
        """
        When the C{cProfile} module is not present,
        L{app.CProfileRunner.run} should raise a C{SystemExit}
        exception and log the C{ImportError}.
        """
        savedModules = sys.modules.copy()
        sys.modules['cProfile'] = None
        config = twistd.ServerOptions()
        config['profiler'] = 'cProfile'
        profiler = app.AppProfiler(config)
        try:
            self.assertRaises(SystemExit, profiler.run, None)
        finally:
            sys.modules.clear()
            sys.modules.update(savedModules)

    def test_unknownProfiler(self):
        """
        Check that L{app.AppProfiler} raises L{SystemExit} when given an
        unknown profiler name.
        """
        config = twistd.ServerOptions()
        config['profile'] = self.mktemp()
        config['profiler'] = 'foobar'
        error = self.assertRaises(SystemExit, app.AppProfiler, config)
        self.assertEqual(str(error), 'Unsupported profiler name: foobar')

    def test_defaultProfiler(self):
        """
        L{app.Profiler} defaults to the cprofile profiler if not specified.
        """
        profiler = app.AppProfiler({})
        self.assertEqual(profiler.profiler, 'cprofile')

    def test_profilerNameCaseInsentive(self):
        """
        The case of the profiler name passed to L{app.AppProfiler} is not
        relevant.
        """
        profiler = app.AppProfiler({'profiler': 'CprOfile'})
        self.assertEqual(profiler.profiler, 'cprofile')