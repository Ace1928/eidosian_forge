import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
class MemoryPool(threadpool.ThreadPool):
    """
    A deterministic threadpool that uses in-memory data structures to queue
    work rather than threads to execute work.
    """

    def __init__(self, coordinator, failTest, newWorker, *args, **kwargs):
        """
        Initialize this L{MemoryPool} with a test case.

        @param coordinator: a worker used to coordinate work in the L{Team}
            underlying this threadpool.
        @type coordinator: L{twisted._threads.IExclusiveWorker}

        @param failTest: A 1-argument callable taking an exception and raising
            a test-failure exception.
        @type failTest: 1-argument callable taking (L{Failure}) and raising
            L{unittest.FailTest}.

        @param newWorker: a 0-argument callable that produces a new
            L{twisted._threads.IWorker} provider on each invocation.
        @type newWorker: 0-argument callable returning
            L{twisted._threads.IWorker}.
        """
        self._coordinator = coordinator
        self._failTest = failTest
        self._newWorker = newWorker
        threadpool.ThreadPool.__init__(self, *args, **kwargs)

    def _pool(self, currentLimit, threadFactory):
        """
        Override testing hook to create a deterministic threadpool.

        @param currentLimit: A 1-argument callable which returns the current
            threadpool size limit.

        @param threadFactory: ignored in this invocation; a 0-argument callable
            that would produce a thread.

        @return: a L{Team} backed by the coordinator and worker passed to
            L{MemoryPool.__init__}.
        """

        def respectLimit():
            stats = team.statistics()
            if stats.busyWorkerCount + stats.idleWorkerCount >= currentLimit():
                return None
            return self._newWorker()
        team = Team(coordinator=self._coordinator, createWorker=respectLimit, logException=self._failTest)
        return team