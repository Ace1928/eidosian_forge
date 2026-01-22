import abc
import logging
import threading
import time
from contextlib import contextmanager
from inspect import getframeinfo, stack
from typing import Any, Dict, List, Optional, Set
def _run_watchdog(self):
    batch_size = max(1, self._request_queue.size())
    timer_requests = self._request_queue.get(batch_size, self._max_interval)
    self.register_timers(timer_requests)
    now = time.time()
    reaped_worker_ids = set()
    for worker_id, expired_timers in self.get_expired_timers(now).items():
        log.info('Reaping worker_id=[%s]. Expired timers: %s', worker_id, self._get_scopes(expired_timers))
        if self._reap_worker_no_throw(worker_id):
            log.info('Successfully reaped worker=[%s]', worker_id)
            reaped_worker_ids.add(worker_id)
        else:
            log.error('Error reaping worker=[%s]. Will retry on next watchdog.', worker_id)
    self.clear_timers(reaped_worker_ids)