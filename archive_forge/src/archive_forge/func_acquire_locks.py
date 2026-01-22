import logging
import os
import signal
import time
import traceback
from datetime import datetime
from enum import Enum
from multiprocessing import Process
from typing import List, Set
from redis import ConnectionPool, Redis
from .connections import parse_connection
from .defaults import DEFAULT_LOGGING_DATE_FORMAT, DEFAULT_LOGGING_FORMAT, DEFAULT_SCHEDULER_FALLBACK_PERIOD
from .job import Job
from .logutils import setup_loghandlers
from .queue import Queue
from .registry import ScheduledJobRegistry
from .serializers import resolve_serializer
from .utils import current_timestamp, parse_names
def acquire_locks(self, auto_start=False):
    """Returns names of queue it successfully acquires lock on"""
    successful_locks = set()
    pid = os.getpid()
    self.log.debug('Trying to acquire locks for %s', ', '.join(self._queue_names))
    for name in self._queue_names:
        if self.connection.set(self.get_locking_key(name), pid, nx=True, ex=self.interval + 60):
            successful_locks.add(name)
    self._scheduled_job_registries = []
    self._acquired_locks = self._acquired_locks.union(successful_locks)
    self.lock_acquisition_time = datetime.now()
    if self._acquired_locks and auto_start:
        if not self._process or not self._process.is_alive():
            self.start()
    return successful_locks