import futurist
from taskflow.conductors.backends import impl_executor
from taskflow.utils import threading_utils as tu
class NonBlockingConductor(impl_executor.ExecutorConductor):
    """Non-blocking conductor that processes job(s) using a thread executor.

    NOTE(harlowja): A custom executor factory can be provided via keyword
                    argument ``executor_factory``, if provided it will be
                    invoked at
                    :py:meth:`~taskflow.conductors.base.Conductor.run` time
                    with one positional argument (this conductor) and it must
                    return a compatible `executor`_ which can be used
                    to submit jobs to. If ``None`` is a provided a thread pool
                    backed executor is selected by default (it will have
                    an equivalent number of workers as this conductors
                    simultaneous job count).

    .. _executor: https://docs.python.org/dev/library/                  concurrent.futures.html#executor-objects
    """
    MAX_SIMULTANEOUS_JOBS = tu.get_optimal_thread_count()
    '\n    Default maximum number of jobs that can be in progress at the same time.\n    '

    def _default_executor_factory(self):
        max_simultaneous_jobs = self._max_simultaneous_jobs
        if max_simultaneous_jobs <= 0:
            max_workers = tu.get_optimal_thread_count()
        else:
            max_workers = max_simultaneous_jobs
        return futurist.ThreadPoolExecutor(max_workers=max_workers)

    def __init__(self, name, jobboard, persistence=None, engine=None, engine_options=None, wait_timeout=None, log=None, max_simultaneous_jobs=MAX_SIMULTANEOUS_JOBS, executor_factory=None):
        super(NonBlockingConductor, self).__init__(name, jobboard, persistence=persistence, engine=engine, engine_options=engine_options, wait_timeout=wait_timeout, log=log, max_simultaneous_jobs=max_simultaneous_jobs)
        if executor_factory is None:
            self._executor_factory = self._default_executor_factory
        else:
            if not callable(executor_factory):
                raise ValueError("Provided keyword argument 'executor_factory' must be callable")
            self._executor_factory = executor_factory