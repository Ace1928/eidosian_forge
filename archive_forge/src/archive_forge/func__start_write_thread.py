from __future__ import annotations
import asyncio
import queue
import sys
import threading
import time
from contextlib import contextmanager
from typing import Generator, TextIO, cast
from .application import get_app_session, run_in_terminal
from .output import Output
def _start_write_thread(self) -> threading.Thread:
    thread = threading.Thread(target=self._write_thread, name='patch-stdout-flush-thread', daemon=True)
    thread.start()
    return thread