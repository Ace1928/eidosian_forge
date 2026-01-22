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
def _open_lockfile(self) -> None:
    if self.stream_lock and (not self.stream_lock.closed):
        self._console_log('Lockfile already open in this process')
        return
    lock_file = self.lockFilename
    with self._alter_umask():
        self.stream_lock = self.atomic_open(lock_file)
    self._do_chown_and_chmod(lock_file)