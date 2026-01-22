import functools
import queue
import threading
from concurrent import futures as _futures
from concurrent.futures import process as _process
from futurist import _green
from futurist import _thread
from futurist import _utils
class GreenThreadPoolExecutor(_futures.Executor):
    """Executor that uses a green thread pool to execute calls asynchronously.

    See: https://docs.python.org/dev/library/concurrent.futures.html
    and http://eventlet.net/doc/modules/greenpool.html for information on
    how this works.

    It gathers statistics about the submissions executed for post-analysis...
    """
    threading = _green.threading

    def __init__(self, max_workers=1000, check_and_reject=None):
        """Initializes a green thread pool executor.

        :param max_workers: maximum number of workers that can be
                            simulatenously active at the same time, further
                            submitted work will be queued up when this limit
                            is reached.
        :type max_workers: int
        :param check_and_reject: a callback function that will be provided
                                 two position arguments, the first argument
                                 will be this executor instance, and the second
                                 will be the number of currently queued work
                                 items in this executors backlog; the callback
                                 should raise a :py:class:`.RejectedSubmission`
                                 exception if it wants to have this submission
                                 rejected.
        :type check_and_reject: callback
        """
        if not _utils.EVENTLET_AVAILABLE:
            raise RuntimeError('Eventlet is needed to use a green executor')
        if max_workers <= 0:
            raise ValueError('Max workers must be greater than zero')
        self._max_workers = max_workers
        self._pool = _green.Pool(self._max_workers)
        self._delayed_work = _green.Queue()
        self._check_and_reject = check_and_reject or (lambda e, waiting: None)
        self._shutdown_lock = self.threading.lock_object()
        self._shutdown = False
        self._gatherer = _Gatherer(self._submit, self.threading.lock_object)

    @property
    def alive(self):
        """Accessor to determine if the executor is alive/active."""
        return not self._shutdown

    @property
    def statistics(self):
        """:class:`.ExecutorStatistics` about the executors executions."""
        return self._gatherer.statistics

    def submit(self, fn, *args, **kwargs):
        """Submit some work to be executed (and gather statistics).

        :param args: non-keyworded arguments
        :type args: list
        :param kwargs: key-value arguments
        :type kwargs: dictionary
        """
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError('Can not schedule new futures after being shutdown')
            self._check_and_reject(self, self._delayed_work.qsize())
            return self._gatherer.submit(fn, *args, **kwargs)

    def _submit(self, fn, *args, **kwargs):
        f = GreenFuture()
        work = _utils.WorkItem(f, fn, args, kwargs)
        if not self._spin_up(work):
            self._delayed_work.put(work)
        return f

    def _spin_up(self, work):
        """Spin up a greenworker if less than max_workers.

        :param work: work to be given to the greenworker
        :returns: whether a green worker was spun up or not
        :rtype: boolean
        """
        alive = self._pool.running() + self._pool.waiting()
        if alive < self._max_workers:
            self._pool.spawn_n(_green.GreenWorker(work, self._delayed_work))
            return True
        return False

    def shutdown(self, wait=True):
        with self._shutdown_lock:
            if not self._shutdown:
                self._shutdown = True
                shutoff = True
            else:
                shutoff = False
        if wait and shutoff:
            self._delayed_work.join()
            self._pool.waitall()