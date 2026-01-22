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
@property
def _conn(self) -> sqlite3.Connection:
    if self._connection is None:
        raise ValueError('no active connection')
    return self._connection