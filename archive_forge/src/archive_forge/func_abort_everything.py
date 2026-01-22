from __future__ import print_function, division, absolute_import
import asyncio
import concurrent.futures
import contextlib
import time
from uuid import uuid4
import weakref
from .parallel import parallel_config
from .parallel import AutoBatchingMixin, ParallelBackendBase
def abort_everything(self, ensure_ready=True):
    """ Tell the client to cancel any task submitted via this instance

        joblib.Parallel will never access those results
        """
    with self.waiting_futures.lock:
        self.waiting_futures.futures.clear()
        while not self.waiting_futures.queue.empty():
            self.waiting_futures.queue.get()