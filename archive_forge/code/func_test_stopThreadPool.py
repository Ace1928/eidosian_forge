import gc
import threading
from weakref import ref
from twisted.internet.interfaces import IReactorThreads
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.threadable import isInIOThread
from twisted.python.threadpool import ThreadPool
from twisted.python.versions import Version
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