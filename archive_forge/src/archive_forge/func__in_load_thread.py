from __future__ import annotations
import datetime
import os
import threading
from abc import ABCMeta, abstractmethod
from asyncio import get_running_loop
from typing import AsyncGenerator, Iterable, Sequence
def _in_load_thread(self) -> None:
    try:
        self._loaded_strings = []
        for item in self.history.load_history_strings():
            with self._lock:
                self._loaded_strings.append(item)
            for event in self._string_load_events:
                event.set()
    finally:
        with self._lock:
            self._loaded = True
        for event in self._string_load_events:
            event.set()