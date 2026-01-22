from __future__ import annotations
import functools
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from watchdog.events import EVENT_TYPE_OPENED, FileSystemEvent, PatternMatchingEventHandler
from watchdog.utils import echo
from watchdog.utils.event_debouncer import EventDebouncer
from watchdog.utils.process_watcher import ProcessWatcher
def _restart_process(self):
    if self._is_trick_stopping:
        return
    self._stop_process()
    self._start_process()
    self.restart_count += 1