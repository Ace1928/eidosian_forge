from __future__ import absolute_import
import logging
import threading
import typing
from typing import Optional
class Heartbeater(object):

    def __init__(self, manager: 'StreamingPullManager', period: int=_DEFAULT_PERIOD):
        self._thread: Optional[threading.Thread] = None
        self._operational_lock = threading.Lock()
        self._manager = manager
        self._stop_event = threading.Event()
        self._period = period

    def heartbeat(self) -> None:
        """Periodically send streaming pull heartbeats."""
        while not self._stop_event.is_set():
            if self._manager.heartbeat():
                _LOGGER.debug('Sent heartbeat.')
            self._stop_event.wait(timeout=self._period)
        _LOGGER.debug('%s exiting.', _HEARTBEAT_WORKER_NAME)

    def start(self) -> None:
        with self._operational_lock:
            if self._thread is not None:
                raise ValueError('Heartbeater is already running.')
            self._stop_event.clear()
            thread = threading.Thread(name=_HEARTBEAT_WORKER_NAME, target=self.heartbeat)
            thread.daemon = True
            thread.start()
            _LOGGER.debug('Started helper thread %s', thread.name)
            self._thread = thread

    def stop(self) -> None:
        with self._operational_lock:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join()
            self._thread = None