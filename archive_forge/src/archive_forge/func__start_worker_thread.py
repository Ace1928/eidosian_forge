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
def _start_worker_thread(app: Celery, concurrency: int=1, pool: str='solo', loglevel: Union[str, int]=WORKER_LOGLEVEL, logfile: Optional[str]=None, WorkController: Any=TestWorkController, perform_ping_check: bool=True, shutdown_timeout: float=10.0, **kwargs) -> Iterable[worker.WorkController]:
    """Start Celery worker in a thread.

    Yields:
        celery.worker.Worker: worker instance.
    """
    setup_app_for_worker(app, loglevel, logfile)
    if perform_ping_check:
        assert 'celery.ping' in app.tasks
    with app.connection(hostname=os.environ.get('TEST_BROKER')) as conn:
        conn.default_channel.queue_declare
    worker = WorkController(app=app, concurrency=concurrency, hostname=anon_nodename(), pool=pool, loglevel=loglevel, logfile=logfile, ready_callback=None, without_heartbeat=kwargs.pop('without_heartbeat', True), without_mingle=True, without_gossip=True, **kwargs)
    t = threading.Thread(target=worker.start, daemon=True)
    t.start()
    worker.ensure_started()
    _set_task_join_will_block(False)
    try:
        yield worker
    finally:
        from celery.worker import state
        state.should_terminate = 0
        t.join(shutdown_timeout)
        if t.is_alive():
            raise RuntimeError('Worker thread failed to exit within the allocated timeout. Consider raising `shutdown_timeout` if your tasks take longer to execute.')
        state.should_terminate = None