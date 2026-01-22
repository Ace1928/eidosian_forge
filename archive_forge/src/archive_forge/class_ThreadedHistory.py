from __future__ import annotations
import datetime
import os
import threading
from abc import ABCMeta, abstractmethod
from asyncio import get_running_loop
from typing import AsyncGenerator, Iterable, Sequence
class ThreadedHistory(History):
    """
    Wrapper around `History` implementations that run the `load()` generator in
    a thread.

    Use this to increase the start-up time of prompt_toolkit applications.
    History entries are available as soon as they are loaded. We don't have to
    wait for everything to be loaded.
    """

    def __init__(self, history: History) -> None:
        super().__init__()
        self.history = history
        self._load_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._string_load_events: list[threading.Event] = []

    async def load(self) -> AsyncGenerator[str, None]:
        """
        Like `History.load(), but call `self.load_history_strings()` in a
        background thread.
        """
        if not self._load_thread:
            self._load_thread = threading.Thread(target=self._in_load_thread, daemon=True)
            self._load_thread.start()
        loop = get_running_loop()
        event = threading.Event()
        event.set()
        self._string_load_events.append(event)
        items_yielded = 0
        try:
            while True:
                got_timeout = await loop.run_in_executor(None, lambda: event.wait(timeout=0.5))
                if not got_timeout:
                    continue

                def in_executor() -> tuple[list[str], bool]:
                    with self._lock:
                        new_items = self._loaded_strings[items_yielded:]
                        done = self._loaded
                        event.clear()
                    return (new_items, done)
                new_items, done = await loop.run_in_executor(None, in_executor)
                items_yielded += len(new_items)
                for item in new_items:
                    yield item
                if done:
                    break
        finally:
            self._string_load_events.remove(event)

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

    def append_string(self, string: str) -> None:
        with self._lock:
            self._loaded_strings.insert(0, string)
        self.store_string(string)

    def load_history_strings(self) -> Iterable[str]:
        return self.history.load_history_strings()

    def store_string(self, string: str) -> None:
        self.history.store_string(string)

    def __repr__(self) -> str:
        return f'ThreadedHistory({self.history!r})'