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
class PersistentScheduler(Scheduler):
    """Scheduler backed by :mod:`shelve` database."""
    persistence = shelve
    known_suffixes = ('', '.db', '.dat', '.bak', '.dir')
    _store = None

    def __init__(self, *args, **kwargs):
        self.schedule_filename = kwargs.get('schedule_filename')
        super().__init__(*args, **kwargs)

    def _remove_db(self):
        for suffix in self.known_suffixes:
            with platforms.ignore_errno(errno.ENOENT):
                os.remove(self.schedule_filename + suffix)

    def _open_schedule(self):
        return self.persistence.open(self.schedule_filename, writeback=True)

    def _destroy_open_corrupted_schedule(self, exc):
        error('Removing corrupted schedule file %r: %r', self.schedule_filename, exc, exc_info=True)
        self._remove_db()
        return self._open_schedule()

    def setup_schedule(self):
        try:
            self._store = self._open_schedule()
            self._store.keys()
        except Exception as exc:
            self._store = self._destroy_open_corrupted_schedule(exc)
        self._create_schedule()
        tz = self.app.conf.timezone
        stored_tz = self._store.get('tz')
        if stored_tz is not None and stored_tz != tz:
            warning('Reset: Timezone changed from %r to %r', stored_tz, tz)
            self._store.clear()
        utc = self.app.conf.enable_utc
        stored_utc = self._store.get('utc_enabled')
        if stored_utc is not None and stored_utc != utc:
            choices = {True: 'enabled', False: 'disabled'}
            warning('Reset: UTC changed from %s to %s', choices[stored_utc], choices[utc])
            self._store.clear()
        entries = self._store.setdefault('entries', {})
        self.merge_inplace(self.app.conf.beat_schedule)
        self.install_default_entries(self.schedule)
        self._store.update({'__version__': __version__, 'tz': tz, 'utc_enabled': utc})
        self.sync()
        debug('Current schedule:\n' + '\n'.join((repr(entry) for entry in entries.values())))

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

    def get_schedule(self):
        return self._store['entries']

    def set_schedule(self, schedule):
        self._store['entries'] = schedule
    schedule = property(get_schedule, set_schedule)

    def sync(self):
        if self._store is not None:
            self._store.sync()

    def close(self):
        self.sync()
        self._store.close()

    @property
    def info(self):
        return f'    . db -> {self.schedule_filename}'