import abc
import futurist
from taskflow import task as ta
from taskflow.types import failure
from taskflow.types import notifier
def _create_executor(self, max_workers=None):
    if max_workers is None:
        max_workers = self.DEFAULT_WORKERS
    return futurist.GreenThreadPoolExecutor(max_workers=max_workers)