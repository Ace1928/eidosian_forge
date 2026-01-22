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
class BaseResultConsumer:
    """Manager responsible for consuming result messages."""

    def __init__(self, backend, app, accept, pending_results, pending_messages):
        self.backend = backend
        self.app = app
        self.accept = accept
        self._pending_results = pending_results
        self._pending_messages = pending_messages
        self.on_message = None
        self.buckets = WeakKeyDictionary()
        self.drainer = drainers[detect_environment()](self)

    def start(self, initial_task_id, **kwargs):
        raise NotImplementedError()

    def stop(self):
        pass

    def drain_events(self, timeout=None):
        raise NotImplementedError()

    def consume_from(self, task_id):
        raise NotImplementedError()

    def cancel_for(self, task_id):
        raise NotImplementedError()

    def _after_fork(self):
        self.buckets.clear()
        self.buckets = WeakKeyDictionary()
        self.on_message = None
        self.on_after_fork()

    def on_after_fork(self):
        pass

    def drain_events_until(self, p, timeout=None, on_interval=None):
        return self.drainer.drain_events_until(p, timeout=timeout, on_interval=on_interval)

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

    def on_wait_for_pending(self, result, timeout=None, **kwargs):
        pass

    def on_out_of_band_result(self, message):
        self.on_state_change(message.payload, message)

    def _get_pending_result(self, task_id):
        for mapping in self._pending_results:
            try:
                return mapping[task_id]
            except KeyError:
                pass
        raise KeyError(task_id)

    def on_state_change(self, meta, message):
        if self.on_message:
            self.on_message(meta)
        if meta['status'] in states.READY_STATES:
            task_id = meta['task_id']
            try:
                result = self._get_pending_result(task_id)
            except KeyError:
                self._pending_messages.put(task_id, meta)
            else:
                result._maybe_set_cache(meta)
                buckets = self.buckets
                try:
                    bucket = buckets.pop(result)
                except KeyError:
                    pass
                else:
                    bucket.append(result)
        sleep(0)