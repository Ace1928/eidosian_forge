from __future__ import division
import os
import sys
from math import sqrt
import functools
import collections
import time
import threading
import itertools
from uuid import uuid4
from numbers import Integral
import warnings
import queue
import weakref
from contextlib import nullcontext
from multiprocessing import TimeoutError
from ._multiprocessing_helpers import mp
from .logger import Logger, short_format_time
from .disk import memstr_to_bytes
from ._parallel_backends import (FallbackToBackend, MultiprocessingBackend,
from ._utils import eval_expr, _Sentinel
from ._parallel_backends import AutoBatchingMixin  # noqa
from ._parallel_backends import ParallelBackendBase  # noqa
class Parallel(Logger):
    """ Helper class for readable parallel mapping.

        Read more in the :ref:`User Guide <parallel>`.

        Parameters
        ----------
        n_jobs: int, default: None
            The maximum number of concurrently running jobs, such as the number
            of Python worker processes when backend="multiprocessing"
            or the size of the thread-pool when backend="threading".
            If -1 all CPUs are used.
            If 1 is given, no parallel computing code is used at all, and the
            behavior amounts to a simple python `for` loop. This mode is not
            compatible with `timeout`.
            For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
            n_jobs = -2, all CPUs but one are used.
            None is a marker for 'unset' that will be interpreted as n_jobs=1
            unless the call is performed under a :func:`~parallel_config`
            context manager that sets another value for ``n_jobs``.
        backend: str, ParallelBackendBase instance or None, default: 'loky'
            Specify the parallelization backend implementation.
            Supported backends are:

            - "loky" used by default, can induce some
              communication and memory overhead when exchanging input and
              output data with the worker Python processes. On some rare
              systems (such as Pyiodide), the loky backend may not be
              available.
            - "multiprocessing" previous process-based backend based on
              `multiprocessing.Pool`. Less robust than `loky`.
            - "threading" is a very low-overhead backend but it suffers
              from the Python Global Interpreter Lock if the called function
              relies a lot on Python objects. "threading" is mostly useful
              when the execution bottleneck is a compiled extension that
              explicitly releases the GIL (for instance a Cython loop wrapped
              in a "with nogil" block or an expensive call to a library such
              as NumPy).
            - finally, you can register backends by calling
              :func:`~register_parallel_backend`. This will allow you to
              implement a backend of your liking.

            It is not recommended to hard-code the backend name in a call to
            :class:`~Parallel` in a library. Instead it is recommended to set
            soft hints (prefer) or hard constraints (require) so as to make it
            possible for library users to change the backend from the outside
            using the :func:`~parallel_config` context manager.
        return_as: str in {'list', 'generator'}, default: 'list'
            If 'list', calls to this instance will return a list, only when
            all results have been processed and retrieved.
            If 'generator', it will return a generator that yields the results
            as soon as they are available, in the order the tasks have been
            submitted with.
            Future releases are planned to also support 'generator_unordered',
            in which case the generator immediately yields available results
            independently of the submission order.
        prefer: str in {'processes', 'threads'} or None, default: None
            Soft hint to choose the default backend if no specific backend
            was selected with the :func:`~parallel_config` context manager.
            The default process-based backend is 'loky' and the default
            thread-based backend is 'threading'. Ignored if the ``backend``
            parameter is specified.
        require: 'sharedmem' or None, default None
            Hard constraint to select the backend. If set to 'sharedmem',
            the selected backend will be single-host and thread-based even
            if the user asked for a non-thread based backend with
            :func:`~joblib.parallel_config`.
        verbose: int, optional
            The verbosity level: if non zero, progress messages are
            printed. Above 50, the output is sent to stdout.
            The frequency of the messages increases with the verbosity level.
            If it more than 10, all iterations are reported.
        timeout: float, optional
            Timeout limit for each task to complete.  If any task takes longer
            a TimeOutError will be raised. Only applied when n_jobs != 1
        pre_dispatch: {'all', integer, or expression, as in '3*n_jobs'}
            The number of batches (of tasks) to be pre-dispatched.
            Default is '2*n_jobs'. When batch_size="auto" this is reasonable
            default and the workers should never starve. Note that only basic
            arithmetics are allowed here and no modules can be used in this
            expression.
        batch_size: int or 'auto', default: 'auto'
            The number of atomic tasks to dispatch at once to each
            worker. When individual evaluations are very fast, dispatching
            calls to workers can be slower than sequential computation because
            of the overhead. Batching fast computations together can mitigate
            this.
            The ``'auto'`` strategy keeps track of the time it takes for a
            batch to complete, and dynamically adjusts the batch size to keep
            the time on the order of half a second, using a heuristic. The
            initial batch size is 1.
            ``batch_size="auto"`` with ``backend="threading"`` will dispatch
            batches of a single task at a time as the threading backend has
            very little overhead and using larger batch size has not proved to
            bring any gain in that case.
        temp_folder: str, optional
            Folder to be used by the pool for memmapping large arrays
            for sharing memory with worker processes. If None, this will try in
            order:

            - a folder pointed by the JOBLIB_TEMP_FOLDER environment
              variable,
            - /dev/shm if the folder exists and is writable: this is a
              RAM disk filesystem available by default on modern Linux
              distributions,
            - the default system temporary folder that can be
              overridden with TMP, TMPDIR or TEMP environment
              variables, typically /tmp under Unix operating systems.

            Only active when backend="loky" or "multiprocessing".
        max_nbytes int, str, or None, optional, 1M by default
            Threshold on the size of arrays passed to the workers that
            triggers automated memory mapping in temp_folder. Can be an int
            in Bytes, or a human-readable string, e.g., '1M' for 1 megabyte.
            Use None to disable memmapping of large arrays.
            Only active when backend="loky" or "multiprocessing".
        mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, default: 'r'
            Memmapping mode for numpy arrays passed to workers. None will
            disable memmapping, other modes defined in the numpy.memmap doc:
            https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
            Also, see 'max_nbytes' parameter documentation for more details.

        Notes
        -----

        This object uses workers to compute in parallel the application of a
        function to many different arguments. The main functionality it brings
        in addition to using the raw multiprocessing or concurrent.futures API
        are (see examples for details):

        * More readable code, in particular since it avoids
          constructing list of arguments.

        * Easier debugging:
            - informative tracebacks even when the error happens on
              the client side
            - using 'n_jobs=1' enables to turn off parallel computing
              for debugging without changing the codepath
            - early capture of pickling errors

        * An optional progress meter.

        * Interruption of multiprocesses jobs with 'Ctrl-C'

        * Flexible pickling control for the communication to and from
          the worker processes.

        * Ability to use shared memory efficiently with worker
          processes for large numpy-based datastructures.

        Note that the intended usage is to run one call at a time. Multiple
        calls to the same Parallel object will result in a ``RuntimeError``

        Examples
        --------

        A simple example:

        >>> from math import sqrt
        >>> from joblib import Parallel, delayed
        >>> Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

        Reshaping the output when the function has several return
        values:

        >>> from math import modf
        >>> from joblib import Parallel, delayed
        >>> r = Parallel(n_jobs=1)(delayed(modf)(i/2.) for i in range(10))
        >>> res, i = zip(*r)
        >>> res
        (0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
        >>> i
        (0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)

        The progress meter: the higher the value of `verbose`, the more
        messages:

        >>> from time import sleep
        >>> from joblib import Parallel, delayed
        >>> r = Parallel(n_jobs=2, verbose=10)(
        ...     delayed(sleep)(.2) for _ in range(10)) #doctest: +SKIP
        [Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    0.6s
        [Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:    0.8s
        [Parallel(n_jobs=2)]: Done  10 out of  10 | elapsed:    1.4s finished

        Traceback example, note how the line of the error is indicated
        as well as the values of the parameter passed to the function that
        triggered the exception, even though the traceback happens in the
        child process:

        >>> from heapq import nlargest
        >>> from joblib import Parallel, delayed
        >>> Parallel(n_jobs=2)(
        ... delayed(nlargest)(2, n) for n in (range(4), 'abcde', 3))
        ... # doctest: +SKIP
        -----------------------------------------------------------------------
        Sub-process traceback:
        -----------------------------------------------------------------------
        TypeError                                      Mon Nov 12 11:37:46 2012
        PID: 12934                                Python 2.7.3: /usr/bin/python
        ........................................................................
        /usr/lib/python2.7/heapq.pyc in nlargest(n=2, iterable=3, key=None)
            419         if n >= size:
            420             return sorted(iterable, key=key, reverse=True)[:n]
            421
            422     # When key is none, use simpler decoration
            423     if key is None:
        --> 424         it = izip(iterable, count(0,-1))           # decorate
            425         result = _nlargest(n, it)
            426         return map(itemgetter(0), result)          # undecorate
            427
            428     # General case, slowest method
         TypeError: izip argument #1 must support iteration
        _______________________________________________________________________


        Using pre_dispatch in a producer/consumer situation, where the
        data is generated on the fly. Note how the producer is first
        called 3 times before the parallel loop is initiated, and then
        called to generate new data on the fly:

        >>> from math import sqrt
        >>> from joblib import Parallel, delayed
        >>> def producer():
        ...     for i in range(6):
        ...         print('Produced %s' % i)
        ...         yield i
        >>> out = Parallel(n_jobs=2, verbose=100, pre_dispatch='1.5*n_jobs')(
        ...     delayed(sqrt)(i) for i in producer()) #doctest: +SKIP
        Produced 0
        Produced 1
        Produced 2
        [Parallel(n_jobs=2)]: Done 1 jobs     | elapsed:  0.0s
        Produced 3
        [Parallel(n_jobs=2)]: Done 2 jobs     | elapsed:  0.0s
        Produced 4
        [Parallel(n_jobs=2)]: Done 3 jobs     | elapsed:  0.0s
        Produced 5
        [Parallel(n_jobs=2)]: Done 4 jobs     | elapsed:  0.0s
        [Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s remaining: 0.0s
        [Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s finished

    """

    def __init__(self, n_jobs=default_parallel_config['n_jobs'], backend=default_parallel_config['backend'], return_as='list', verbose=default_parallel_config['verbose'], timeout=None, pre_dispatch='2 * n_jobs', batch_size='auto', temp_folder=default_parallel_config['temp_folder'], max_nbytes=default_parallel_config['max_nbytes'], mmap_mode=default_parallel_config['mmap_mode'], prefer=default_parallel_config['prefer'], require=default_parallel_config['require']):
        super().__init__()
        if n_jobs is None:
            n_jobs = default_parallel_config['n_jobs']
        active_backend, context_config = _get_active_backend(prefer=prefer, require=require, verbose=verbose)
        nesting_level = active_backend.nesting_level
        self.verbose = _get_config_param(verbose, context_config, 'verbose')
        self.timeout = timeout
        self.pre_dispatch = pre_dispatch
        if return_as not in {'list', 'generator'}:
            raise ValueError(f'Expected `return_as` parameter to be a string equal to "list" or "generator", but got {return_as} instead')
        self.return_as = return_as
        self.return_generator = return_as != 'list'
        self._backend_args = {k: _get_config_param(param, context_config, k) for param, k in [(max_nbytes, 'max_nbytes'), (temp_folder, 'temp_folder'), (mmap_mode, 'mmap_mode'), (prefer, 'prefer'), (require, 'require'), (verbose, 'verbose')]}
        if isinstance(self._backend_args['max_nbytes'], str):
            self._backend_args['max_nbytes'] = memstr_to_bytes(self._backend_args['max_nbytes'])
        self._backend_args['verbose'] = max(0, self._backend_args['verbose'] - 50)
        if DEFAULT_MP_CONTEXT is not None:
            self._backend_args['context'] = DEFAULT_MP_CONTEXT
        elif hasattr(mp, 'get_context'):
            self._backend_args['context'] = mp.get_context()
        if backend is default_parallel_config['backend'] or backend is None:
            backend = active_backend
        elif isinstance(backend, ParallelBackendBase):
            if backend.nesting_level is None:
                backend.nesting_level = nesting_level
        elif hasattr(backend, 'Pool') and hasattr(backend, 'Lock'):
            self._backend_args['context'] = backend
            backend = MultiprocessingBackend(nesting_level=nesting_level)
        elif backend not in BACKENDS and backend in MAYBE_AVAILABLE_BACKENDS:
            warnings.warn(f"joblib backend '{backend}' is not available on your system, falling back to {DEFAULT_BACKEND}.", UserWarning, stacklevel=2)
            BACKENDS[backend] = BACKENDS[DEFAULT_BACKEND]
            backend = BACKENDS[DEFAULT_BACKEND](nesting_level=nesting_level)
        else:
            try:
                backend_factory = BACKENDS[backend]
            except KeyError as e:
                raise ValueError('Invalid backend: %s, expected one of %r' % (backend, sorted(BACKENDS.keys()))) from e
            backend = backend_factory(nesting_level=nesting_level)
        n_jobs = _get_config_param(n_jobs, context_config, 'n_jobs')
        if n_jobs is None:
            n_jobs = backend.default_n_jobs
        self.n_jobs = n_jobs
        if require == 'sharedmem' and (not getattr(backend, 'supports_sharedmem', False)):
            raise ValueError('Backend %s does not support shared memory' % backend)
        if batch_size == 'auto' or (isinstance(batch_size, Integral) and batch_size > 0):
            self.batch_size = batch_size
        else:
            raise ValueError("batch_size must be 'auto' or a positive integer, got: %r" % batch_size)
        if not isinstance(backend, SequentialBackend):
            if self.return_generator and (not backend.supports_return_generator):
                raise ValueError('Backend {} does not support return_as={}'.format(backend, return_as))
            self._lock = threading.RLock()
            self._jobs = collections.deque()
            self._pending_outputs = list()
            self._ready_batches = queue.Queue()
            self._reducer_callback = None
        self._backend = backend
        self._running = False
        self._managed_backend = False
        self._id = uuid4().hex
        self._call_ref = None

    def __enter__(self):
        self._managed_backend = True
        self._calling = False
        self._initialize_backend()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._managed_backend = False
        if self.return_generator and self._calling:
            self._abort()
        self._terminate_and_reset()

    def _initialize_backend(self):
        """Build a process or thread pool and return the number of workers"""
        try:
            n_jobs = self._backend.configure(n_jobs=self.n_jobs, parallel=self, **self._backend_args)
            if self.timeout is not None and (not self._backend.supports_timeout):
                warnings.warn("The backend class {!r} does not support timeout. You have set 'timeout={}' in Parallel but the 'timeout' parameter will not be used.".format(self._backend.__class__.__name__, self.timeout))
        except FallbackToBackend as e:
            self._backend = e.backend
            n_jobs = self._initialize_backend()
        return n_jobs

    def _effective_n_jobs(self):
        if self._backend:
            return self._backend.effective_n_jobs(self.n_jobs)
        return 1

    def _terminate_and_reset(self):
        if hasattr(self._backend, 'stop_call') and self._calling:
            self._backend.stop_call()
        self._calling = False
        if not self._managed_backend:
            self._backend.terminate()

    def _dispatch(self, batch):
        """Queue the batch for computing, with or without multiprocessing

        WARNING: this method is not thread-safe: it should be only called
        indirectly via dispatch_one_batch.

        """
        if self._aborting:
            return
        batch_size = len(batch)
        self.n_dispatched_tasks += batch_size
        self.n_dispatched_batches += 1
        dispatch_timestamp = time.time()
        batch_tracker = BatchCompletionCallBack(dispatch_timestamp, batch_size, self)
        self._jobs.append(batch_tracker)
        job = self._backend.apply_async(batch, callback=batch_tracker)
        batch_tracker.register_job(job)

    def dispatch_next(self):
        """Dispatch more data for parallel processing

        This method is meant to be called concurrently by the multiprocessing
        callback. We rely on the thread-safety of dispatch_one_batch to protect
        against concurrent consumption of the unprotected iterator.

        """
        if not self.dispatch_one_batch(self._original_iterator):
            self._iterating = False
            self._original_iterator = None

    def dispatch_one_batch(self, iterator):
        """Prefetch the tasks for the next batch and dispatch them.

        The effective size of the batch is computed here.
        If there are no more jobs to dispatch, return False, else return True.

        The iterator consumption and dispatching is protected by the same
        lock so calling this function should be thread safe.

        """
        if self._aborting:
            return False
        batch_size = self._get_batch_size()
        with self._lock:
            try:
                tasks = self._ready_batches.get(block=False)
            except queue.Empty:
                n_jobs = self._cached_effective_n_jobs
                big_batch_size = batch_size * n_jobs
                islice = list(itertools.islice(iterator, big_batch_size))
                if len(islice) == 0:
                    return False
                elif iterator is self._original_iterator and len(islice) < big_batch_size:
                    final_batch_size = max(1, len(islice) // (10 * n_jobs))
                else:
                    final_batch_size = max(1, len(islice) // n_jobs)
                for i in range(0, len(islice), final_batch_size):
                    tasks = BatchedCalls(islice[i:i + final_batch_size], self._backend.get_nested_backend(), self._reducer_callback, self._pickle_cache)
                    self._ready_batches.put(tasks)
                tasks = self._ready_batches.get(block=False)
            if len(tasks) == 0:
                return False
            else:
                self._dispatch(tasks)
                return True

    def _get_batch_size(self):
        """Returns the effective batch size for dispatch"""
        if self.batch_size == 'auto':
            return self._backend.compute_batch_size()
        else:
            return self.batch_size

    def _print(self, msg):
        """Display the message on stout or stderr depending on verbosity"""
        if not self.verbose:
            return
        if self.verbose < 50:
            writer = sys.stderr.write
        else:
            writer = sys.stdout.write
        writer(f'[{self}]: {msg}\n')

    def _is_completed(self):
        """Check if all tasks have been completed"""
        return self.n_completed_tasks == self.n_dispatched_tasks and (not (self._iterating or self._aborting))

    def print_progress(self):
        """Display the process of the parallel execution only a fraction
           of time, controlled by self.verbose.
        """
        if not self.verbose:
            return
        elapsed_time = time.time() - self._start_time
        if self._is_completed():
            self._print(f'Done {self.n_completed_tasks:3d} out of {self.n_completed_tasks:3d} | elapsed: {short_format_time(elapsed_time)} finished')
            return
        elif self._original_iterator is not None:
            if _verbosity_filter(self.n_dispatched_batches, self.verbose):
                return
            self._print(f'Done {self.n_completed_tasks:3d} tasks      | elapsed: {short_format_time(elapsed_time)}')
        else:
            index = self.n_completed_tasks
            total_tasks = self.n_dispatched_tasks
            if not index == 0:
                cursor = total_tasks - index + 1 - self._pre_dispatch_amount
                frequency = total_tasks // self.verbose + 1
                is_last_item = index + 1 == total_tasks
                if is_last_item or cursor % frequency:
                    return
            remaining_time = elapsed_time / index * (self.n_dispatched_tasks - index * 1.0)
            self._print(f'Done {index:3d} out of {total_tasks:3d} | elapsed: {short_format_time(elapsed_time)} remaining: {short_format_time(remaining_time)}')

    def _abort(self):
        self._aborting = True
        backend = self._backend
        if not self._aborted and hasattr(backend, 'abort_everything'):
            ensure_ready = self._managed_backend
            backend.abort_everything(ensure_ready=ensure_ready)
        self._aborted = True

    def _start(self, iterator, pre_dispatch):
        self._iterating = False
        if self.dispatch_one_batch(iterator):
            self._iterating = self._original_iterator is not None
        while self.dispatch_one_batch(iterator):
            pass
        if pre_dispatch == 'all':
            self._iterating = False

    def _get_outputs(self, iterator, pre_dispatch):
        """Iterator returning the tasks' output as soon as they are ready."""
        dispatch_thread_id = threading.get_ident()
        detach_generator_exit = False
        try:
            self._start(iterator, pre_dispatch)
            yield
            with self._backend.retrieval_context():
                yield from self._retrieve()
        except GeneratorExit:
            self._exception = True
            if dispatch_thread_id != threading.get_ident():
                if not IS_PYPY:
                    warnings.warn("A generator produced by joblib.Parallel has been gc'ed in an unexpected thread. This behavior should not cause major -issues but to make sure, please report this warning and your use case at https://github.com/joblib/joblib/issues so it can be investigated.")
                detach_generator_exit = True
                _parallel = self

                class _GeneratorExitThread(threading.Thread):

                    def run(self):
                        _parallel._abort()
                        if _parallel.return_generator:
                            _parallel._warn_exit_early()
                        _parallel._terminate_and_reset()
                _GeneratorExitThread(name='GeneratorExitThread').start()
                return
            self._abort()
            if self.return_generator:
                self._warn_exit_early()
            raise
        except BaseException:
            self._exception = True
            self._abort()
            raise
        finally:
            _remaining_outputs = [] if self._exception else self._jobs
            self._jobs = collections.deque()
            self._running = False
            if not detach_generator_exit:
                self._terminate_and_reset()
        while len(_remaining_outputs) > 0:
            batched_results = _remaining_outputs.popleft()
            batched_results = batched_results.get_result(self.timeout)
            for result in batched_results:
                yield result

    def _wait_retrieval(self):
        """Return True if we need to continue retriving some tasks."""
        if self._iterating:
            return True
        if self.n_completed_tasks < self.n_dispatched_tasks:
            return True
        if not self._backend.supports_retrieve_callback:
            if len(self._jobs) > 0:
                return True
        return False

    def _retrieve(self):
        while self._wait_retrieval():
            if self._aborting:
                self._raise_error_fast()
                break
            if len(self._jobs) == 0 or self._jobs[0].get_status(timeout=self.timeout) == TASK_PENDING:
                time.sleep(0.01)
                continue
            with self._lock:
                batched_results = self._jobs.popleft()
            batched_results = batched_results.get_result(self.timeout)
            for result in batched_results:
                self._nb_consumed += 1
                yield result

    def _raise_error_fast(self):
        """If we are aborting, raise if a job caused an error."""
        with self._lock:
            error_job = next((job for job in self._jobs if job.status == TASK_ERROR), None)
        if error_job is not None:
            error_job.get_result(self.timeout)

    def _warn_exit_early(self):
        """Warn the user if the generator is gc'ed before being consumned."""
        ready_outputs = self.n_completed_tasks - self._nb_consumed
        is_completed = self._is_completed()
        msg = ''
        if ready_outputs:
            msg += f'{ready_outputs} tasks have been successfully executed  but not used.'
            if not is_completed:
                msg += ' Additionally, '
        if not is_completed:
            msg += f'{self.n_dispatched_tasks - self.n_completed_tasks} tasks which were still being processed by the workers have been cancelled.'
        if msg:
            msg += ' You could benefit from adjusting the input task iterator to limit unnecessary computation time.'
            warnings.warn(msg)

    def _get_sequential_output(self, iterable):
        """Separate loop for sequential output.

        This simplifies the traceback in case of errors and reduces the
        overhead of calling sequential tasks with `joblib`.
        """
        try:
            self._iterating = True
            self._original_iterator = iterable
            batch_size = self._get_batch_size()
            if batch_size != 1:
                it = iter(iterable)
                iterable_batched = iter(lambda: tuple(itertools.islice(it, batch_size)), ())
                iterable = (task for batch in iterable_batched for task in batch)
            yield None
            for func, args, kwargs in iterable:
                self.n_dispatched_batches += 1
                self.n_dispatched_tasks += 1
                res = func(*args, **kwargs)
                self.n_completed_tasks += 1
                self.print_progress()
                yield res
                self._nb_consumed += 1
        except BaseException:
            self._exception = True
            self._aborting = True
            self._aborted = True
            raise
        finally:
            self.print_progress()
            self._running = False
            self._iterating = False
            self._original_iterator = None

    def _reset_run_tracking(self):
        """Reset the counters and flags used to track the execution."""
        with getattr(self, '_lock', nullcontext()):
            if self._running:
                msg = 'This Parallel instance is already running !'
                if self.return_generator is True:
                    msg += ' Before submitting new tasks, you must wait for the completion of all the previous tasks, or clean all references to the output generator.'
                raise RuntimeError(msg)
            self._running = True
        self.n_dispatched_batches = 0
        self.n_dispatched_tasks = 0
        self.n_completed_tasks = 0
        self._nb_consumed = 0
        self._exception = False
        self._aborting = False
        self._aborted = False

    def __call__(self, iterable):
        """Main function to dispatch parallel tasks."""
        self._reset_run_tracking()
        self._start_time = time.time()
        if not self._managed_backend:
            n_jobs = self._initialize_backend()
        else:
            n_jobs = self._effective_n_jobs()
        if n_jobs == 1:
            output = self._get_sequential_output(iterable)
            next(output)
            return output if self.return_generator else list(output)
        with self._lock:
            self._call_id = uuid4().hex
        self._cached_effective_n_jobs = n_jobs
        if isinstance(self._backend, LokyBackend):

            def _batched_calls_reducer_callback():
                self._backend._workers._temp_folder_manager.set_current_context(self._id)
            self._reducer_callback = _batched_calls_reducer_callback
        self._cached_effective_n_jobs = n_jobs
        backend_name = self._backend.__class__.__name__
        if n_jobs == 0:
            raise RuntimeError('%s has no active worker.' % backend_name)
        self._print(f'Using backend {backend_name} with {n_jobs} concurrent workers.')
        if hasattr(self._backend, 'start_call'):
            self._backend.start_call()
        self._calling = True
        iterator = iter(iterable)
        pre_dispatch = self.pre_dispatch
        if pre_dispatch == 'all':
            self._original_iterator = None
            self._pre_dispatch_amount = 0
        else:
            self._original_iterator = iterator
            if hasattr(pre_dispatch, 'endswith'):
                pre_dispatch = eval_expr(pre_dispatch.replace('n_jobs', str(n_jobs)))
            self._pre_dispatch_amount = pre_dispatch = int(pre_dispatch)
            iterator = itertools.islice(iterator, self._pre_dispatch_amount)
        self._pickle_cache = dict()
        output = self._get_outputs(iterator, pre_dispatch)
        self._call_ref = weakref.ref(output)
        next(output)
        return output if self.return_generator else list(output)

    def __repr__(self):
        return '%s(n_jobs=%s)' % (self.__class__.__name__, self.n_jobs)