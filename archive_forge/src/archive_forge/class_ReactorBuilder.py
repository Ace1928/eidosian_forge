import os
import signal
import time
from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Type, Union, cast
from zope.interface import Interface
from twisted.python import log
from twisted.python.deprecate import _fullyQualifiedName as fullyQualifiedName
from twisted.python.failure import Failure
from twisted.python.reflect import namedAny
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, SynchronousTestCase
from twisted.trial.util import DEFAULT_TIMEOUT_DURATION, acquireAttribute
class ReactorBuilder:
    """
    L{SynchronousTestCase} mixin which provides a reactor-creation API.  This
    mixin defines C{setUp} and C{tearDown}, so mix it in before
    L{SynchronousTestCase} or call its methods from the overridden ones in the
    subclass.

    @cvar skippedReactors: A dict mapping FQPN strings of reactors for
        which the tests defined by this class will be skipped to strings
        giving the skip message.
    @cvar requiredInterfaces: A C{list} of interfaces which the reactor must
        provide or these tests will be skipped.  The default, L{None}, means
        that no interfaces are required.
    @ivar reactorFactory: A no-argument callable which returns the reactor to
        use for testing.
    @ivar originalHandler: The SIGCHLD handler which was installed when setUp
        ran and which will be re-installed when tearDown runs.
    @ivar _reactors: A list of FQPN strings giving the reactors for which
        L{SynchronousTestCase}s will be created.
    """
    _reactors = ['twisted.internet.selectreactor.SelectReactor']
    if platform.isWindows():
        _reactors.extend(['twisted.internet.gireactor.PortableGIReactor', 'twisted.internet.win32eventreactor.Win32Reactor', 'twisted.internet.iocpreactor.reactor.IOCPReactor'])
    else:
        _reactors.extend(['twisted.internet.gireactor.GIReactor'])
        _reactors.append('twisted.internet.test.reactormixins.AsyncioSelectorReactor')
        if platform.isMacOSX():
            _reactors.append('twisted.internet.cfreactor.CFReactor')
        else:
            _reactors.extend(['twisted.internet.pollreactor.PollReactor', 'twisted.internet.epollreactor.EPollReactor'])
            if not platform.isLinux():
                _reactors.extend(['twisted.internet.kqreactor.KQueueReactor'])
    reactorFactory: Optional[Callable[[], object]] = None
    originalHandler = None
    requiredInterfaces: Optional[Sequence[Type[Interface]]] = None
    skippedReactors: Dict[str, str] = {}

    def setUp(self):
        """
        Clear the SIGCHLD handler, if there is one, to ensure an environment
        like the one which exists prior to a call to L{reactor.run}.
        """
        if not platform.isWindows():
            self.originalHandler = signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    def tearDown(self):
        """
        Restore the original SIGCHLD handler and reap processes as long as
        there seem to be any remaining.
        """
        if self.originalHandler is not None:
            signal.signal(signal.SIGCHLD, self.originalHandler)
        if process is not None:
            begin = time.time()
            while process.reapProcessHandlers:
                log.msg('ReactorBuilder.tearDown reaping some processes %r' % (process.reapProcessHandlers,))
                process.reapAllProcesses()
                time.sleep(0.001)
                if time.time() - begin > 60:
                    for pid in process.reapProcessHandlers:
                        os.kill(pid, signal.SIGKILL)
                    raise Exception('Timeout waiting for child processes to exit: %r' % (process.reapProcessHandlers,))

    def _unbuildReactor(self, reactor):
        """
        Clean up any resources which may have been allocated for the given
        reactor by its creation or by a test which used it.
        """
        reactor._uninstallHandler()
        if getattr(reactor, '_internalReaders', None) is not None:
            for reader in reactor._internalReaders:
                reactor.removeReader(reader)
                reader.connectionLost(None)
            reactor._internalReaders.clear()
        reactor.disconnectAll()
        calls = reactor.getDelayedCalls()
        for c in calls:
            c.cancel()
        from twisted.internet import reactor as globalReactor
        globalReactor.__dict__ = reactor._originalReactorDict
        globalReactor.__class__ = reactor._originalReactorClass

    def buildReactor(self):
        """
        Create and return a reactor using C{self.reactorFactory}.
        """
        try:
            from twisted.internet import reactor as globalReactor
            from twisted.internet.cfreactor import CFReactor
        except ImportError:
            pass
        else:
            if isinstance(globalReactor, CFReactor) and self.reactorFactory is CFReactor:
                raise SkipTest("CFReactor uses APIs which manipulate global state, so it's not safe to run its own reactor-builder tests under itself")
        try:
            assert self.reactorFactory is not None
            reactor = self.reactorFactory()
            reactor._originalReactorDict = globalReactor.__dict__
            reactor._originalReactorClass = globalReactor.__class__
            globalReactor.__dict__ = reactor.__dict__
            globalReactor.__class__ = reactor.__class__
        except BaseException:
            log.err(None, 'Failed to install reactor')
            self.flushLoggedErrors()
            raise SkipTest(Failure().getErrorMessage())
        else:
            if self.requiredInterfaces is not None:
                missing = [required for required in self.requiredInterfaces if not required.providedBy(reactor)]
                if missing:
                    self._unbuildReactor(reactor)
                    raise SkipTest('%s does not provide %s' % (fullyQualifiedName(reactor.__class__), ','.join([fullyQualifiedName(x) for x in missing])))
        self.addCleanup(self._unbuildReactor, reactor)
        return reactor

    def getTimeout(self):
        """
        Determine how long to run the test before considering it failed.

        @return: A C{int} or C{float} giving a number of seconds.
        """
        return acquireAttribute(self._parents, 'timeout', DEFAULT_TIMEOUT_DURATION)

    def runReactor(self, reactor, timeout=None):
        """
        Run the reactor for at most the given amount of time.

        @param reactor: The reactor to run.

        @type timeout: C{int} or C{float}
        @param timeout: The maximum amount of time, specified in seconds, to
            allow the reactor to run.  If the reactor is still running after
            this much time has elapsed, it will be stopped and an exception
            raised.  If L{None}, the default test method timeout imposed by
            Trial will be used.  This depends on the L{IReactorTime}
            implementation of C{reactor} for correct operation.

        @raise TestTimeoutError: If the reactor is still running after
            C{timeout} seconds.
        """
        if timeout is None:
            timeout = self.getTimeout()
        timedOut = []

        def stop():
            timedOut.append(None)
            reactor.stop()
        timedOutCall = reactor.callLater(timeout, stop)
        reactor.run()
        if timedOut:
            raise TestTimeoutError(f'reactor still running after {timeout} seconds')
        else:
            timedOutCall.cancel()

    @classmethod
    def makeTestCaseClasses(cls: Type['ReactorBuilder']) -> Dict[str, Union[Type['ReactorBuilder'], Type[SynchronousTestCase]]]:
        """
        Create a L{SynchronousTestCase} subclass which mixes in C{cls} for each
        known reactor and return a dict mapping their names to them.
        """
        classes: Dict[str, Union[Type['ReactorBuilder'], Type[SynchronousTestCase]]] = {}
        for reactor in cls._reactors:
            shortReactorName = reactor.split('.')[-1]
            name = (cls.__name__ + '.' + shortReactorName + 'Tests').replace('.', '_')

            class testcase(cls, SynchronousTestCase):
                __module__ = cls.__module__
                if reactor in cls.skippedReactors:
                    skip = cls.skippedReactors[reactor]
                try:
                    reactorFactory = namedAny(reactor)
                except BaseException:
                    skip = Failure().getErrorMessage()
            testcase.__name__ = name
            testcase.__qualname__ = '.'.join(cls.__qualname__.split()[0:-1] + [name])
            classes[testcase.__name__] = testcase
        return classes