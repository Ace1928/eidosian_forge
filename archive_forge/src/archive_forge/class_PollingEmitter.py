from __future__ import with_statement
import os
import threading
from functools import partial
from wandb_watchdog.utils import stat as default_stat
from wandb_watchdog.utils.dirsnapshot import DirectorySnapshot, DirectorySnapshotDiff
from wandb_watchdog.observers.api import (
from wandb_watchdog.events import (
class PollingEmitter(EventEmitter):
    """
    Platform-independent emitter that polls a directory to detect file
    system changes.
    """

    def __init__(self, event_queue, watch, timeout=DEFAULT_EMITTER_TIMEOUT, stat=default_stat, listdir=os.listdir):
        EventEmitter.__init__(self, event_queue, watch, timeout)
        self._snapshot = None
        self._lock = threading.Lock()
        self._take_snapshot = lambda: DirectorySnapshot(self.watch.path, self.watch.is_recursive, stat=stat, listdir=listdir)

    def on_thread_start(self):
        self._snapshot = self._take_snapshot()

    def queue_events(self, timeout):
        if self.stopped_event.wait(timeout):
            return
        with self._lock:
            if not self.should_keep_running():
                return
            try:
                new_snapshot = self._take_snapshot()
            except OSError:
                self.queue_event(DirDeletedEvent(self.watch.path))
                self.stop()
                return
            events = DirectorySnapshotDiff(self._snapshot, new_snapshot)
            self._snapshot = new_snapshot
            for src_path in events.files_deleted:
                self.queue_event(FileDeletedEvent(src_path))
            for src_path in events.files_modified:
                self.queue_event(FileModifiedEvent(src_path))
            for src_path in events.files_created:
                self.queue_event(FileCreatedEvent(src_path))
            for src_path, dest_path in events.files_moved:
                self.queue_event(FileMovedEvent(src_path, dest_path))
            for src_path in events.dirs_deleted:
                self.queue_event(DirDeletedEvent(src_path))
            for src_path in events.dirs_modified:
                self.queue_event(DirModifiedEvent(src_path))
            for src_path in events.dirs_created:
                self.queue_event(DirCreatedEvent(src_path))
            for src_path, dest_path in events.dirs_moved:
                self.queue_event(DirMovedEvent(src_path, dest_path))