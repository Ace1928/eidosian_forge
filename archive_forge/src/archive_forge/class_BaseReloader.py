import itertools
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import weakref
from collections import defaultdict
from functools import lru_cache, wraps
from pathlib import Path
from types import ModuleType
from zipimport import zipimporter
import django
from django.apps import apps
from django.core.signals import request_finished
from django.dispatch import Signal
from django.utils.functional import cached_property
from django.utils.version import get_version_tuple
class BaseReloader:

    def __init__(self):
        self.extra_files = set()
        self.directory_globs = defaultdict(set)
        self._stop_condition = threading.Event()

    def watch_dir(self, path, glob):
        path = Path(path)
        try:
            path = path.absolute()
        except FileNotFoundError:
            logger.debug('Unable to watch directory %s as it cannot be resolved.', path, exc_info=True)
            return
        logger.debug('Watching dir %s with glob %s.', path, glob)
        self.directory_globs[path].add(glob)

    def watched_files(self, include_globs=True):
        """
        Yield all files that need to be watched, including module files and
        files within globs.
        """
        yield from iter_all_python_module_files()
        yield from self.extra_files
        if include_globs:
            for directory, patterns in self.directory_globs.items():
                for pattern in patterns:
                    yield from directory.glob(pattern)

    def wait_for_apps_ready(self, app_reg, django_main_thread):
        """
        Wait until Django reports that the apps have been loaded. If the given
        thread has terminated before the apps are ready, then a SyntaxError or
        other non-recoverable error has been raised. In that case, stop waiting
        for the apps_ready event and continue processing.

        Return True if the thread is alive and the ready event has been
        triggered, or False if the thread is terminated while waiting for the
        event.
        """
        while django_main_thread.is_alive():
            if app_reg.ready_event.wait(timeout=0.1):
                return True
        else:
            logger.debug('Main Django thread has terminated before apps are ready.')
            return False

    def run(self, django_main_thread):
        logger.debug('Waiting for apps ready_event.')
        self.wait_for_apps_ready(apps, django_main_thread)
        from django.urls import get_resolver
        try:
            get_resolver().urlconf_module
        except Exception:
            pass
        logger.debug('Apps ready_event triggered. Sending autoreload_started signal.')
        autoreload_started.send(sender=self)
        self.run_loop()

    def run_loop(self):
        ticker = self.tick()
        while not self.should_stop:
            try:
                next(ticker)
            except StopIteration:
                break
        self.stop()

    def tick(self):
        """
        This generator is called in a loop from run_loop. It's important that
        the method takes care of pausing or otherwise waiting for a period of
        time. This split between run_loop() and tick() is to improve the
        testability of the reloader implementations by decoupling the work they
        do from the loop.
        """
        raise NotImplementedError('subclasses must implement tick().')

    @classmethod
    def check_availability(cls):
        raise NotImplementedError('subclasses must implement check_availability().')

    def notify_file_changed(self, path):
        results = file_changed.send(sender=self, file_path=path)
        logger.debug('%s notified as changed. Signal results: %s.', path, results)
        if not any((res[1] for res in results)):
            trigger_reload(path)

    @property
    def should_stop(self):
        return self._stop_condition.is_set()

    def stop(self):
        self._stop_condition.set()