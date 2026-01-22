import abc
import futurist
from taskflow import task as ta
from taskflow.types import failure
from taskflow.types import notifier
class ParallelGreenThreadTaskExecutor(ParallelThreadTaskExecutor):
    """Executes tasks in parallel using a greenthread pool executor."""
    DEFAULT_WORKERS = 1000
    "\n    Default number of workers when ``None`` is passed; being that\n    greenthreads don't map to native threads or processors very well this\n    is more of a guess/somewhat arbitrary, but it does match what the eventlet\n    greenpool default size is (so at least it's consistent with what eventlet\n    does).\n    "

    def _create_executor(self, max_workers=None):
        if max_workers is None:
            max_workers = self.DEFAULT_WORKERS
        return futurist.GreenThreadPoolExecutor(max_workers=max_workers)