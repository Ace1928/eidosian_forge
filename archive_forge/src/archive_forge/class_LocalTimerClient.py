import logging
import multiprocessing as mp
import os
import signal
import time
from queue import Empty
from typing import Any, Dict, List, Set, Tuple
from .api import RequestQueue, TimerClient, TimerRequest, TimerServer
class LocalTimerClient(TimerClient):
    """
    Client side of ``LocalTimerServer``. This client is meant to be used
    on the same host that the ``LocalTimerServer`` is running on and uses
    pid to uniquely identify a worker. This is particularly useful in situations
    where one spawns a subprocess (trainer) per GPU on a host with multiple
    GPU devices.
    """

    def __init__(self, mp_queue):
        super().__init__()
        self._mp_queue = mp_queue

    def acquire(self, scope_id, expiration_time):
        pid = os.getpid()
        acquire_request = TimerRequest(pid, scope_id, expiration_time)
        self._mp_queue.put(acquire_request)

    def release(self, scope_id):
        pid = os.getpid()
        release_request = TimerRequest(pid, scope_id, -1)
        self._mp_queue.put(release_request)