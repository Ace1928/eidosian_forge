import asyncio
import logging
import sqlite3
from functools import partial
from pathlib import Path
from queue import Empty, Queue, SimpleQueue
from threading import Thread
from typing import (
from warnings import warn
from .context import contextmanager
from .cursor import Cursor
def _stop_running(self):
    self._running = False
    self._tx.put_nowait(_STOP_RUNNING_SENTINEL)