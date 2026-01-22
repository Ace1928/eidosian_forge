from __future__ import annotations
import os
import threading
from typing import Callable, Final, cast
from blinker import ANY, Signal
from watchdog import events
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch
from streamlit.logger import get_logger
from streamlit.util import repr_
from streamlit.watcher import util
class _MultiPathWatcher:
    """Watches multiple paths."""
    _singleton: _MultiPathWatcher | None = None

    @classmethod
    def get_singleton(cls) -> _MultiPathWatcher:
        """Return the singleton _MultiPathWatcher object.

        Instantiates one if necessary.
        """
        if cls._singleton is None:
            _LOGGER.debug('No singleton. Registering one.')
            _MultiPathWatcher()
        return cast('_MultiPathWatcher', _MultiPathWatcher._singleton)

    def __new__(cls) -> _MultiPathWatcher:
        """Constructor."""
        if _MultiPathWatcher._singleton is not None:
            raise RuntimeError('Use .get_singleton() instead')
        return super().__new__(cls)

    def __init__(self) -> None:
        """Constructor."""
        _MultiPathWatcher._singleton = self
        self._folder_handlers: dict[str, _FolderEventHandler] = {}
        self._lock = threading.Lock()
        self._observer = Observer()
        self._observer.start()

    def __repr__(self) -> str:
        return repr_(self)

    def watch_path(self, path: str, callback: Callable[[str], None], *, glob_pattern: str | None=None, allow_nonexistent: bool=False) -> None:
        """Start watching a path."""
        folder_path = os.path.abspath(os.path.dirname(path))
        with self._lock:
            folder_handler = self._folder_handlers.get(folder_path)
            if folder_handler is None:
                folder_handler = _FolderEventHandler()
                self._folder_handlers[folder_path] = folder_handler
                folder_handler.watch = self._observer.schedule(folder_handler, folder_path, recursive=True)
            folder_handler.add_path_change_listener(path, callback, glob_pattern=glob_pattern, allow_nonexistent=allow_nonexistent)

    def stop_watching_path(self, path: str, callback: Callable[[str], None]) -> None:
        """Stop watching a path."""
        folder_path = os.path.abspath(os.path.dirname(path))
        with self._lock:
            folder_handler = self._folder_handlers.get(folder_path)
            if folder_handler is None:
                _LOGGER.debug('Cannot stop watching path, because it is already not being watched. %s', folder_path)
                return
            folder_handler.remove_path_change_listener(path, callback)
            if not folder_handler.is_watching_paths():
                self._observer.unschedule(folder_handler.watch)
                del self._folder_handlers[folder_path]

    def close(self) -> None:
        with self._lock:
            'Close this _MultiPathWatcher object forever.'
            if len(self._folder_handlers) != 0:
                self._folder_handlers = {}
                _LOGGER.debug('Stopping observer thread even though there is a non-zero number of event observers!')
            else:
                _LOGGER.debug('Stopping observer thread')
            self._observer.stop()
            self._observer.join(timeout=5)