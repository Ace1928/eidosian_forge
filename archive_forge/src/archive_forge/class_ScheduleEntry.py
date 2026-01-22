import copy
import errno
import heapq
import os
import shelve
import sys
import time
import traceback
from calendar import timegm
from collections import namedtuple
from functools import total_ordering
from threading import Event, Thread
from billiard import ensure_multiprocessing
from billiard.common import reset_signals
from billiard.context import Process
from kombu.utils.functional import maybe_evaluate, reprcall
from kombu.utils.objects import cached_property
from . import __version__, platforms, signals
from .exceptions import reraise
from .schedules import crontab, maybe_schedule
from .utils.functional import is_numeric_value
from .utils.imports import load_extension_class_names, symbol_by_name
from .utils.log import get_logger, iter_open_logger_fds
from .utils.time import humanize_seconds, maybe_make_aware
@total_ordering
class ScheduleEntry:
    """An entry in the scheduler.

    Arguments:
        name (str): see :attr:`name`.
        schedule (~celery.schedules.schedule): see :attr:`schedule`.
        args (Tuple): see :attr:`args`.
        kwargs (Dict): see :attr:`kwargs`.
        options (Dict): see :attr:`options`.
        last_run_at (~datetime.datetime): see :attr:`last_run_at`.
        total_run_count (int): see :attr:`total_run_count`.
        relative (bool): Is the time relative to when the server starts?
    """
    name = None
    schedule = None
    args = None
    kwargs = None
    options = None
    last_run_at = None
    total_run_count = 0

    def __init__(self, name=None, task=None, last_run_at=None, total_run_count=None, schedule=None, args=(), kwargs=None, options=None, relative=False, app=None):
        self.app = app
        self.name = name
        self.task = task
        self.args = args
        self.kwargs = kwargs if kwargs else {}
        self.options = options if options else {}
        self.schedule = maybe_schedule(schedule, relative, app=self.app)
        self.last_run_at = last_run_at or self.default_now()
        self.total_run_count = total_run_count or 0

    def default_now(self):
        return self.schedule.now() if self.schedule else self.app.now()
    _default_now = default_now

    def _next_instance(self, last_run_at=None):
        """Return new instance, with date and count fields updated."""
        return self.__class__(**dict(self, last_run_at=last_run_at or self.default_now(), total_run_count=self.total_run_count + 1))
    __next__ = next = _next_instance

    def __reduce__(self):
        return (self.__class__, (self.name, self.task, self.last_run_at, self.total_run_count, self.schedule, self.args, self.kwargs, self.options))

    def update(self, other):
        """Update values from another entry.

        Will only update "editable" fields:
            ``task``, ``schedule``, ``args``, ``kwargs``, ``options``.
        """
        self.__dict__.update({'task': other.task, 'schedule': other.schedule, 'args': other.args, 'kwargs': other.kwargs, 'options': other.options})

    def is_due(self):
        """See :meth:`~celery.schedules.schedule.is_due`."""
        return self.schedule.is_due(self.last_run_at)

    def __iter__(self):
        return iter(vars(self).items())

    def __repr__(self):
        return '<{name}: {0.name} {call} {0.schedule}'.format(self, call=reprcall(self.task, self.args or (), self.kwargs or {}), name=type(self).__name__)

    def __lt__(self, other):
        if isinstance(other, ScheduleEntry):
            return id(self) < id(other)
        return NotImplemented

    def editable_fields_equal(self, other):
        for attr in ('task', 'args', 'kwargs', 'options', 'schedule'):
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def __eq__(self, other):
        """Test schedule entries equality.

        Will only compare "editable" fields:
        ``task``, ``schedule``, ``args``, ``kwargs``, ``options``.
        """
        return self.editable_fields_equal(other)