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
@text_factory.setter
def text_factory(self, factory: Callable[[bytes], Any]) -> None:
    self._conn.text_factory = factory