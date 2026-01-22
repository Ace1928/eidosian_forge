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
class _FolderEventHandler(events.FileSystemEventHandler):
    """Listen to folder events. If certain paths change, fire a callback.

    The super class, FileSystemEventHandler, listens to changes to *folders*,
    but we need to listen to changes to *both* folders and files. I believe
    this is a limitation of the Mac FSEvents system API, and the watchdog
    library takes the lower common denominator.

    So in this class we watch for folder events and then filter them based
    on whether or not we care for the path the event is about.
    """

    def __init__(self) -> None:
        super().__init__()
        self._watched_paths: dict[str, WatchedPath] = {}
        self._lock = threading.Lock()
        self.watch: ObservedWatch | None = None

    def __repr__(self) -> str:
        return repr_(self)

    def add_path_change_listener(self, path: str, callback: Callable[[str], None], *, glob_pattern: str | None=None, allow_nonexistent: bool=False) -> None:
        """Add a path to this object's event filter."""
        with self._lock:
            watched_path = self._watched_paths.get(path, None)
            if watched_path is None:
                md5 = util.calc_md5_with_blocking_retries(path, glob_pattern=glob_pattern, allow_nonexistent=allow_nonexistent)
                modification_time = util.path_modification_time(path, allow_nonexistent)
                watched_path = WatchedPath(md5=md5, modification_time=modification_time, glob_pattern=glob_pattern, allow_nonexistent=allow_nonexistent)
                self._watched_paths[path] = watched_path
            watched_path.on_changed.connect(callback, weak=False)

    def remove_path_change_listener(self, path: str, callback: Callable[[str], None]) -> None:
        """Remove a path from this object's event filter."""
        with self._lock:
            watched_path = self._watched_paths.get(path, None)
            if watched_path is None:
                return
            watched_path.on_changed.disconnect(callback)
            if not watched_path.on_changed.has_receivers_for(ANY):
                del self._watched_paths[path]

    def is_watching_paths(self) -> bool:
        """Return true if this object has 1+ paths in its event filter."""
        return len(self._watched_paths) > 0

    def handle_path_change_event(self, event: events.FileSystemEvent) -> None:
        """Handle when a path (corresponding to a file or dir) is changed.

        The events that can call this are modification, creation or moved
        events.
        """
        if event.event_type == events.EVENT_TYPE_MODIFIED:
            changed_path = event.src_path
        elif event.event_type == events.EVENT_TYPE_MOVED:
            event = cast(events.FileSystemMovedEvent, event)
            _LOGGER.debug('Move event: src %s; dest %s', event.src_path, event.dest_path)
            changed_path = event.dest_path
        elif event.event_type == events.EVENT_TYPE_CREATED:
            changed_path = event.src_path
        else:
            _LOGGER.debug("Don't care about event type %s", event.event_type)
            return
        changed_path = os.path.abspath(changed_path)
        changed_path_info = self._watched_paths.get(changed_path, None)
        if changed_path_info is None:
            _LOGGER.debug('Ignoring changed path %s.\nWatched_paths: %s', changed_path, self._watched_paths)
            return
        modification_time = util.path_modification_time(changed_path, changed_path_info.allow_nonexistent)
        if modification_time != 0.0 and modification_time == changed_path_info.modification_time:
            _LOGGER.debug('File/dir timestamp did not change: %s', changed_path)
            return
        changed_path_info.modification_time = modification_time
        new_md5 = util.calc_md5_with_blocking_retries(changed_path, glob_pattern=changed_path_info.glob_pattern, allow_nonexistent=changed_path_info.allow_nonexistent)
        if new_md5 == changed_path_info.md5:
            _LOGGER.debug('File/dir MD5 did not change: %s', changed_path)
            return
        _LOGGER.debug('File/dir MD5 changed: %s', changed_path)
        changed_path_info.md5 = new_md5
        changed_path_info.on_changed.send(changed_path)

    def on_created(self, event: events.FileSystemEvent) -> None:
        self.handle_path_change_event(event)

    def on_modified(self, event: events.FileSystemEvent) -> None:
        self.handle_path_change_event(event)

    def on_moved(self, event: events.FileSystemEvent) -> None:
        self.handle_path_change_event(event)