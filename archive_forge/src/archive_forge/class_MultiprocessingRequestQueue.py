import logging
import multiprocessing as mp
import os
import signal
import time
from queue import Empty
from typing import Any, Dict, List, Set, Tuple
from .api import RequestQueue, TimerClient, TimerRequest, TimerServer
class MultiprocessingRequestQueue(RequestQueue):
    """
    A ``RequestQueue`` backed by python ``multiprocessing.Queue``
    """

    def __init__(self, mp_queue: mp.Queue):
        super().__init__()
        self._mp_queue = mp_queue

    def size(self) -> int:
        return self._mp_queue.qsize()

    def get(self, size, timeout: float) -> List[TimerRequest]:
        requests = []
        wait = timeout
        for _ in range(0, size):
            start = time.time()
            try:
                r = self._mp_queue.get(block=True, timeout=wait)
            except Empty:
                break
            requests.append(r)
            wait = wait - (time.time() - start)
            if wait <= 0:
                break
        return requests