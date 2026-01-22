import datetime
import errno
import logging
import os
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from io import TextIOWrapper
from logging.handlers import BaseRotatingHandler, TimedRotatingFileHandler
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Tuple
from portalocker import LOCK_EX, lock, unlock
import logging.handlers  # noqa: E402
def _do_lock(self) -> None:
    if self.is_locked:
        return
    self._open_lockfile()
    if self.stream_lock:
        for _i in range(self.maxLockAttempts):
            try:
                lock(self.stream_lock, LOCK_EX)
                self.is_locked = True
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f'Cannot acquire lock after {self.maxLockAttempts} attempts')
    else:
        self._console_log('No self.stream_lock to lock', stack=True)