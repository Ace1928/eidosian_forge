import abc
import contextlib
import functools
import itertools
import threading
from oslo_utils import excutils
from oslo_utils import timeutils
from taskflow.conductors import base
from taskflow import exceptions as excp
from taskflow.listeners import logging as logging_listener
from taskflow import logging
from taskflow import states
from taskflow.types import timing as tt
from taskflow.utils import iter_utils
from taskflow.utils import misc
def _on_job_done(self, job, fut):
    consume = False
    trash = False
    try:
        consume = fut.result()
    except KeyboardInterrupt:
        with excutils.save_and_reraise_exception():
            self._log.warn('Job dispatching interrupted: %s', job)
    except Exception:
        self._log.warn('Job dispatching failed: %s', job, exc_info=True)
        trash = True
    try:
        self._try_finish_job(job, consume, trash)
    finally:
        self._dispatched.discard(fut)