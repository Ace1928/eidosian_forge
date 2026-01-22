import gc
import threading
from weakref import ref
from twisted.internet.interfaces import IReactorThreads
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.threadable import isInIOThread
from twisted.python.threadpool import ThreadPool
from twisted.python.versions import Version
class ThreadTestsBuilder(ReactorBuilder):
    """
    Builder for defining tests relating to L{IReactorThreads}.
    """
    requiredInterfaces = (IReactorThreads,)

    def test_getThreadPool(self):
        """
        C{reactor.getThreadPool()} returns an instance of L{ThreadPool} which
        starts when C{reactor.run()} is called and stops before it returns.
        """
        state = []
        reactor = self.buildReactor()
        pool = reactor.getThreadPool()
        self.assertIsInstance(pool, ThreadPool)
        self.assertFalse(pool.started, 'Pool should not start before reactor.run')

        def f():
            state.append(pool.started)
            state.append(pool.joined)
            reactor.stop()
        reactor.callWhenRunning(f)
        self.runReactor(reactor, 2)
        self.assertTrue(state[0], 'Pool should start after reactor.run')
        self.assertFalse(state[1], 'Pool should not be joined before reactor.stop')
        self.assertTrue(pool.joined, 'Pool should be stopped after reactor.run returns')

    def test_suggestThreadPoolSize(self):
        """
        C{reactor.suggestThreadPoolSize()} sets the maximum size of the reactor
        threadpool.
        """
        reactor = self.buildReactor()
        reactor.suggestThreadPoolSize(17)
        pool = reactor.getThreadPool()
        self.assertEqual(pool.max, 17)

    def test_delayedCallFromThread(self):
        """
        A function scheduled with L{IReactorThreads.callFromThread} invoked
        from a delayed call is run immediately in the next reactor iteration.

        When invoked from the reactor thread, previous implementations of
        L{IReactorThreads.callFromThread} would skip the pipe/socket based wake
        up step, assuming the reactor would wake up on its own.  However, this
        resulted in the reactor not noticing an insert into the thread queue at
        the right time (in this case, after the thread queue has been processed
        for that reactor iteration).
        """
        reactor = self.buildReactor()

        def threadCall():
            reactor.stop()
        reactor.callLater(0, reactor.callFromThread, threadCall)
        before = reactor.seconds()
        self.runReactor(reactor, 60)
        after = reactor.seconds()
        self.assertTrue(after - before < 30)

    def test_callFromThread(self):
        """
        A function scheduled with L{IReactorThreads.callFromThread} invoked
        from another thread is run in the reactor thread.
        """
        reactor = self.buildReactor()
        result = []

        def threadCall():
            result.append(threading.current_thread())
            reactor.stop()
        reactor.callLater(0, reactor.callInThread, reactor.callFromThread, threadCall)
        self.runReactor(reactor, 5)
        self.assertEqual(result, [threading.current_thread()])

    def test_stopThreadPool(self):
        """
        When the reactor stops, L{ReactorBase._stopThreadPool} drops the
        reactor's direct reference to its internal threadpool and removes
        the associated startup and shutdown triggers.

        This is the case of the thread pool being created before the reactor
        is run.
        """
        reactor = self.buildReactor()
        threadpool = ref(reactor.getThreadPool())
        reactor.callWhenRunning(reactor.stop)
        self.runReactor(reactor)
        gc.collect()
        self.assertIsNone(threadpool())

    def test_stopThreadPoolWhenStartedAfterReactorRan(self):
        """
        We must handle the case of shutting down the thread pool when it was
        started after the reactor was run in a special way.

        Some implementation background: The thread pool is started with
        callWhenRunning, which only returns a system trigger ID when it is
        invoked before the reactor is started.

        This is the case of the thread pool being created after the reactor
        is started.
        """
        reactor = self.buildReactor()
        threadPoolRefs = []

        def acquireThreadPool():
            threadPoolRefs.append(ref(reactor.getThreadPool()))
            reactor.stop()
        reactor.callWhenRunning(acquireThreadPool)
        self.runReactor(reactor)
        gc.collect()
        self.assertIsNone(threadPoolRefs[0]())

    def test_cleanUpThreadPoolEvenBeforeReactorIsRun(self):
        """
        When the reactor has its shutdown event fired before it is run, the
        thread pool is completely destroyed.

        For what it's worth, the reason we support this behavior at all is
        because Trial does this.

        This is the case of the thread pool being created without the reactor
        being started at al.
        """
        reactor = self.buildReactor()
        threadPoolRef = ref(reactor.getThreadPool())
        reactor.fireSystemEvent('shutdown')
        gc.collect()
        self.assertIsNone(threadPoolRef())

    def test_isInIOThread(self):
        """
        The reactor registers itself as the I/O thread when it runs so that
        L{twisted.python.threadable.isInIOThread} returns C{True} if it is
        called in the thread the reactor is running in.
        """
        results = []
        reactor = self.buildReactor()

        def check():
            results.append(isInIOThread())
            reactor.stop()
        reactor.callWhenRunning(check)
        self.runReactor(reactor)
        self.assertEqual([True], results)

    def test_isNotInIOThread(self):
        """
        The reactor registers itself as the I/O thread when it runs so that
        L{twisted.python.threadable.isInIOThread} returns C{False} if it is
        called in a different thread than the reactor is running in.
        """
        results = []
        reactor = self.buildReactor()

        def check():
            results.append(isInIOThread())
            reactor.callFromThread(reactor.stop)
        reactor.callInThread(check)
        self.runReactor(reactor)
        self.assertEqual([False], results)

    def test_threadPoolCurrentThreadDeprecated(self):
        self.callDeprecated(version=(Version('Twisted', 22, 1, 0), 'threading.current_thread'), f=ThreadPool.currentThread)