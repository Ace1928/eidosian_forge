from typing import Callable
from zope.interface import implementer
from ._convenience import Quit
from ._ithreads import IExclusiveWorker
@implementer(IExclusiveWorker)
class LockWorker:
    """
    An L{IWorker} implemented based on a mutual-exclusion lock.
    """

    def __init__(self, lock, local):
        """
        @param lock: A mutual-exclusion lock, with C{acquire} and C{release}
            methods.
        @type lock: L{threading.Lock}

        @param local: Local storage.
        @type local: L{threading.local}
        """
        self._quit = Quit()
        self._lock = lock
        self._local = local

    def do(self, work: Callable[[], None]) -> None:
        """
        Do the given work on this thread, with the mutex acquired.  If this is
        called re-entrantly, return and wait for the outer invocation to do the
        work.

        @param work: the work to do with the lock held.
        """
        lock = self._lock
        local = self._local
        self._quit.check()
        working = getattr(local, 'working', None)
        if working is None:
            working = local.working = []
            working.append(work)
            lock.acquire()
            try:
                while working:
                    working.pop(0)()
            finally:
                lock.release()
                local.working = None
        else:
            working.append(work)

    def quit(self):
        """
        Quit this L{LockWorker}.
        """
        self._quit.set()
        self._lock = None