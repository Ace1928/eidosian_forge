import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterable, Optional, Union
import celery.worker.consumer  # noqa
from celery import Celery, worker
from celery.result import _set_task_join_will_block, allow_join_result
from celery.utils.dispatch import Signal
from celery.utils.nodenames import anon_nodename
class TestWorkController(worker.WorkController):
    """Worker that can synchronize on being fully started."""
    logger_queue = None

    def __init__(self, *args, **kwargs):
        self._on_started = threading.Event()
        super().__init__(*args, **kwargs)
        if self.pool_cls.__module__.split('.')[-1] == 'prefork':
            from billiard import Queue
            self.logger_queue = Queue()
            self.pid = os.getpid()
            try:
                from tblib import pickling_support
                pickling_support.install()
            except ImportError:
                pass
            self.queue_listener = logging.handlers.QueueListener(self.logger_queue, logging.getLogger())
            self.queue_listener.start()

    class QueueHandler(logging.handlers.QueueHandler):

        def prepare(self, record):
            record.from_queue = True
            return record

        def handleError(self, record):
            if logging.raiseExceptions:
                raise

    def start(self):
        if self.logger_queue:
            handler = self.QueueHandler(self.logger_queue)
            handler.addFilter(lambda r: r.process != self.pid and (not getattr(r, 'from_queue', False)))
            logger = logging.getLogger()
            logger.addHandler(handler)
        return super().start()

    def on_consumer_ready(self, consumer):
        """Callback called when the Consumer blueprint is fully started."""
        self._on_started.set()
        test_worker_started.send(sender=self.app, worker=self, consumer=consumer)

    def ensure_started(self):
        """Wait for worker to be fully up and running.

        Warning:
            Worker must be started within a thread for this to work,
            or it will block forever.
        """
        self._on_started.wait()