import gc
import threading
from weakref import ref
from twisted.internet.interfaces import IReactorThreads
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.threadable import isInIOThread
from twisted.python.threadpool import ThreadPool
from twisted.python.versions import Version
def test_threadPoolCurrentThreadDeprecated(self):
    self.callDeprecated(version=(Version('Twisted', 22, 1, 0), 'threading.current_thread'), f=ThreadPool.currentThread)