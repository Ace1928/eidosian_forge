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
@contextmanager
def _start_worker_process(app, concurrency=1, pool='solo', loglevel=WORKER_LOGLEVEL, logfile=None, **kwargs):
    """Start worker in separate process.

    Yields:
        celery.app.worker.Worker: worker instance.
    """
    from celery.apps.multi import Cluster, Node
    app.set_current()
    cluster = Cluster([Node('testworker1@%h')])
    cluster.start()
    try:
        yield
    finally:
        cluster.stopwait()