import logging
import sys
from datetime import datetime
from time import monotonic, time
from weakref import ref
from billiard.common import TERM_SIGNAME
from billiard.einfo import ExceptionWithTraceback
from kombu.utils.encoding import safe_repr, safe_str
from kombu.utils.objects import cached_property
from celery import current_app, signals
from celery.app.task import Context
from celery.app.trace import fast_trace_task, trace_task, trace_task_ret
from celery.concurrency.base import BasePool
from celery.exceptions import (Ignore, InvalidTaskError, Reject, Retry, TaskRevokedError, Terminated,
from celery.platforms import signals as _signals
from celery.utils.functional import maybe, maybe_list, noop
from celery.utils.log import get_logger
from celery.utils.nodenames import gethostname
from celery.utils.serialization import get_pickled_exception
from celery.utils.time import maybe_iso8601, maybe_make_aware, timezone
from . import state
def _announce_cancelled(self):
    task_ready(self)
    self.send_event('task-cancelled')
    reason = 'cancelled by Celery'
    exc = Retry(message=reason)
    self.task.backend.mark_as_retry(self.id, exc, request=self._context)
    self.task.on_retry(exc, self.id, self.args, self.kwargs, None)
    self._already_cancelled = True
    send_retry(self.task, request=self._context, einfo=None)