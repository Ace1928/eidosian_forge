import socket
import threading
import time
from collections import deque
from queue import Empty
from time import sleep
from weakref import WeakKeyDictionary
from kombu.utils.compat import detect_environment
from celery import states
from celery.exceptions import TimeoutError
from celery.utils.threads import THREAD_TIMEOUT_MAX
def _wait_for_pending(self, result, timeout=None, on_interval=None, on_message=None, **kwargs):
    self.on_wait_for_pending(result, timeout=timeout, **kwargs)
    prev_on_m, self.on_message = (self.on_message, on_message)
    try:
        for _ in self.drain_events_until(result.on_ready, timeout=timeout, on_interval=on_interval):
            yield
            sleep(0)
    except socket.timeout:
        raise TimeoutError('The operation timed out.')
    finally:
        self.on_message = prev_on_m