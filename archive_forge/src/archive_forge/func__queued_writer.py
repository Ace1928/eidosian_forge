import functools
import json
import multiprocessing
import os
import threading
from contextlib import contextmanager
from threading import Thread
from ._colorizer import Colorizer
from ._locks_machinery import create_handler_lock
def _queued_writer(self):
    message = None
    queue = self._queue
    lock = self._queue_lock
    while True:
        try:
            message = queue.get()
        except Exception:
            with lock:
                self._error_interceptor.print(None)
            continue
        if message is None:
            break
        if message is True:
            self._confirmation_event.set()
            continue
        with lock:
            try:
                self._sink.write(message)
            except Exception:
                self._error_interceptor.print(message.record)