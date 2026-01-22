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
def _create_schedule(self):
    for _ in (1, 2):
        try:
            self._store['entries']
        except (KeyError, UnicodeDecodeError, TypeError):
            try:
                self._store['entries'] = {}
            except (KeyError, UnicodeDecodeError, TypeError) as exc:
                self._store = self._destroy_open_corrupted_schedule(exc)
                continue
        else:
            if '__version__' not in self._store:
                warning('DB Reset: Account for new __version__ field')
                self._store.clear()
            elif 'tz' not in self._store:
                warning('DB Reset: Account for new tz field')
                self._store.clear()
            elif 'utc_enabled' not in self._store:
                warning('DB Reset: Account for new utc_enabled field')
                self._store.clear()
        break