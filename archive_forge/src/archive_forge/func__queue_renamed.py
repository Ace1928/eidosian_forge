from __future__ import with_statement
from wandb_watchdog.utils import platform
import threading
import errno
import sys
import stat
import os
from wandb_watchdog.observers.api import (
from wandb_watchdog.utils.dirsnapshot import DirectorySnapshot
from wandb_watchdog.events import (
def _queue_renamed(self, src_path, is_directory, ref_snapshot, new_snapshot):
    """
        Compares information from two directory snapshots (one taken before
        the rename operation and another taken right after) to determine the
        destination path of the file system object renamed, and adds
        appropriate events to the event queue.
        """
    try:
        ref_stat_info = ref_snapshot.stat_info(src_path)
    except KeyError:
        if is_directory:
            self.queue_event(DirCreatedEvent(src_path))
            self.queue_event(DirDeletedEvent(src_path))
        else:
            self.queue_event(FileCreatedEvent(src_path))
            self.queue_event(FileDeletedEvent(src_path))
        return
    try:
        dest_path = absolute_path(new_snapshot.path_for_inode(ref_stat_info.st_ino))
        if is_directory:
            event = DirMovedEvent(src_path, dest_path)
            if self.watch.is_recursive:
                for sub_event in event.sub_moved_events():
                    self.queue_event(sub_event)
            self.queue_event(event)
        else:
            self.queue_event(FileMovedEvent(src_path, dest_path))
    except KeyError:
        if is_directory:
            self.queue_event(DirDeletedEvent(src_path))
        else:
            self.queue_event(FileDeletedEvent(src_path))