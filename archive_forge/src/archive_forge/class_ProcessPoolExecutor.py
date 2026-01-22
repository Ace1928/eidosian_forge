import os
from concurrent.futures import _base
import queue
import multiprocessing as mp
import multiprocessing.connection
from multiprocessing.queues import Queue
import threading
import weakref
from functools import partial
import itertools
import sys
from traceback import format_exception
class ProcessPoolExecutor(_base.Executor):

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), *, max_tasks_per_child=None):
        """Initializes a new ProcessPoolExecutor instance.

        Args:
            max_workers: The maximum number of processes that can be used to
                execute the given calls. If None or not given then as many
                worker processes will be created as the machine has processors.
            mp_context: A multiprocessing context to launch the workers. This
                object should provide SimpleQueue, Queue and Process. Useful
                to allow specific multiprocessing start methods.
            initializer: A callable used to initialize worker processes.
            initargs: A tuple of arguments to pass to the initializer.
            max_tasks_per_child: The maximum number of tasks a worker process
                can complete before it will exit and be replaced with a fresh
                worker process. The default of None means worker process will
                live as long as the executor. Requires a non-'fork' mp_context
                start method. When given, we default to using 'spawn' if no
                mp_context is supplied.
        """
        _check_system_limits()
        if max_workers is None:
            self._max_workers = os.cpu_count() or 1
            if sys.platform == 'win32':
                self._max_workers = min(_MAX_WINDOWS_WORKERS, self._max_workers)
        else:
            if max_workers <= 0:
                raise ValueError('max_workers must be greater than 0')
            elif sys.platform == 'win32' and max_workers > _MAX_WINDOWS_WORKERS:
                raise ValueError(f'max_workers must be <= {_MAX_WINDOWS_WORKERS}')
            self._max_workers = max_workers
        if mp_context is None:
            if max_tasks_per_child is not None:
                mp_context = mp.get_context('spawn')
            else:
                mp_context = mp.get_context()
        self._mp_context = mp_context
        self._safe_to_dynamically_spawn_children = self._mp_context.get_start_method(allow_none=False) != 'fork'
        if initializer is not None and (not callable(initializer)):
            raise TypeError('initializer must be a callable')
        self._initializer = initializer
        self._initargs = initargs
        if max_tasks_per_child is not None:
            if not isinstance(max_tasks_per_child, int):
                raise TypeError('max_tasks_per_child must be an integer')
            elif max_tasks_per_child <= 0:
                raise ValueError('max_tasks_per_child must be >= 1')
            if self._mp_context.get_start_method(allow_none=False) == 'fork':
                raise ValueError("max_tasks_per_child is incompatible with the 'fork' multiprocessing start method; supply a different mp_context.")
        self._max_tasks_per_child = max_tasks_per_child
        self._executor_manager_thread = None
        self._processes = {}
        self._shutdown_thread = False
        self._shutdown_lock = threading.Lock()
        self._idle_worker_semaphore = threading.Semaphore(0)
        self._broken = False
        self._queue_count = 0
        self._pending_work_items = {}
        self._cancel_pending_futures = False
        self._executor_manager_thread_wakeup = _ThreadWakeup()
        queue_size = self._max_workers + EXTRA_QUEUED_CALLS
        self._call_queue = _SafeQueue(max_size=queue_size, ctx=self._mp_context, pending_work_items=self._pending_work_items, shutdown_lock=self._shutdown_lock, thread_wakeup=self._executor_manager_thread_wakeup)
        self._call_queue._ignore_epipe = True
        self._result_queue = mp_context.SimpleQueue()
        self._work_ids = queue.Queue()

    def _start_executor_manager_thread(self):
        if self._executor_manager_thread is None:
            if not self._safe_to_dynamically_spawn_children:
                self._launch_processes()
            self._executor_manager_thread = _ExecutorManagerThread(self)
            self._executor_manager_thread.start()
            _threads_wakeups[self._executor_manager_thread] = self._executor_manager_thread_wakeup

    def _adjust_process_count(self):
        if self._idle_worker_semaphore.acquire(blocking=False):
            return
        process_count = len(self._processes)
        if process_count < self._max_workers:
            self._spawn_process()

    def _launch_processes(self):
        assert not self._executor_manager_thread, 'Processes cannot be fork()ed after the thread has started, deadlock in the child processes could result.'
        for _ in range(len(self._processes), self._max_workers):
            self._spawn_process()

    def _spawn_process(self):
        p = self._mp_context.Process(target=_process_worker, args=(self._call_queue, self._result_queue, self._initializer, self._initargs, self._max_tasks_per_child))
        p.start()
        self._processes[p.pid] = p

    def submit(self, fn, /, *args, **kwargs):
        with self._shutdown_lock:
            if self._broken:
                raise BrokenProcessPool(self._broken)
            if self._shutdown_thread:
                raise RuntimeError('cannot schedule new futures after shutdown')
            if _global_shutdown:
                raise RuntimeError('cannot schedule new futures after interpreter shutdown')
            f = _base.Future()
            w = _WorkItem(f, fn, args, kwargs)
            self._pending_work_items[self._queue_count] = w
            self._work_ids.put(self._queue_count)
            self._queue_count += 1
            self._executor_manager_thread_wakeup.wakeup()
            if self._safe_to_dynamically_spawn_children:
                self._adjust_process_count()
            self._start_executor_manager_thread()
            return f
    submit.__doc__ = _base.Executor.submit.__doc__

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the process pool.
                If set to one, the items in the list will be sent one at a time.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        if chunksize < 1:
            raise ValueError('chunksize must be >= 1.')
        results = super().map(partial(_process_chunk, fn), _get_chunks(*iterables, chunksize=chunksize), timeout=timeout)
        return _chain_from_iterable_of_lists(results)

    def shutdown(self, wait=True, *, cancel_futures=False):
        with self._shutdown_lock:
            self._cancel_pending_futures = cancel_futures
            self._shutdown_thread = True
            if self._executor_manager_thread_wakeup is not None:
                self._executor_manager_thread_wakeup.wakeup()
        if self._executor_manager_thread is not None and wait:
            self._executor_manager_thread.join()
        self._executor_manager_thread = None
        self._call_queue = None
        if self._result_queue is not None and wait:
            self._result_queue.close()
        self._result_queue = None
        self._processes = None
        self._executor_manager_thread_wakeup = None
    shutdown.__doc__ = _base.Executor.shutdown.__doc__