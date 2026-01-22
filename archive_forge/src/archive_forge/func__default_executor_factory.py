import futurist
from taskflow.conductors.backends import impl_executor
from taskflow.utils import threading_utils as tu
def _default_executor_factory(self):
    max_simultaneous_jobs = self._max_simultaneous_jobs
    if max_simultaneous_jobs <= 0:
        max_workers = tu.get_optimal_thread_count()
    else:
        max_workers = max_simultaneous_jobs
    return futurist.ThreadPoolExecutor(max_workers=max_workers)