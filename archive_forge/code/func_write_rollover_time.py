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
def write_rollover_time(self) -> None:
    """Write the next rollover time (current value of self.rolloverAt) to the lock file."""
    lock_file = self.clh.stream_lock
    if not lock_file or not self.clh.is_locked:
        self._console_log('No rollover time (lock) file to write to. Lock is not held?')
        return
    lock_file.seek(0)
    lock_file.write(str(self.rolloverAt))
    lock_file.truncate()
    lock_file.flush()
    os.fsync(lock_file.fileno())
    self._console_log(f'Wrote rollover time: {self.rolloverAt}')