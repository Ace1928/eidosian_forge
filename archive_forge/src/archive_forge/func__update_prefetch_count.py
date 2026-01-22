from the broker, processing the messages and keeping the broker connections
import errno
import logging
import os
import warnings
from collections import defaultdict
from time import sleep
from billiard.common import restart_state
from billiard.exceptions import RestartFreqExceeded
from kombu.asynchronous.semaphore import DummyLock
from kombu.exceptions import ContentDisallowed, DecodeError
from kombu.utils.compat import _detect_environment
from kombu.utils.encoding import safe_repr
from kombu.utils.limits import TokenBucket
from vine import ppartial, promise
from celery import bootsteps, signals
from celery.app.trace import build_tracer
from celery.exceptions import (CPendingDeprecationWarning, InvalidTaskError, NotRegistered, WorkerShutdown,
from celery.utils.functional import noop
from celery.utils.log import get_logger
from celery.utils.nodenames import gethostname
from celery.utils.objects import Bunch
from celery.utils.text import truncate
from celery.utils.time import humanize_seconds, rate
from celery.worker import loops
from celery.worker.state import active_requests, maybe_shutdown, requests, reserved_requests, task_reserved
def _update_prefetch_count(self, index=0):
    """Update prefetch count after pool/shrink grow operations.

        Index must be the change in number of processes as a positive
        (increasing) or negative (decreasing) number.

        Note:
            Currently pool grow operations will end up with an offset
            of +1 if the initial size of the pool was 0 (e.g.
            :option:`--autoscale=1,0 <celery worker --autoscale>`).
        """
    num_processes = self.pool.num_processes
    if not self.initial_prefetch_count or not num_processes:
        return
    self.initial_prefetch_count = self.pool.num_processes * self.prefetch_multiplier
    return self._update_qos_eventually(index)