import abc
import futurist
from taskflow import task as ta
from taskflow.types import failure
from taskflow.types import notifier
class ParallelThreadTaskExecutor(ParallelTaskExecutor):
    """Executes tasks in parallel using a thread pool executor."""

    def _create_executor(self, max_workers=None):
        return futurist.ThreadPoolExecutor(max_workers=max_workers)