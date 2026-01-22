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
def _queue_events_except_renames_and_dir_modifications(self, event_list):
    """
        Queues events from the kevent list returned from the call to
        :meth:`select.kqueue.control`.

        .. NOTE:: Queues only the deletions, file modifications,
                  attribute modifications. The other events, namely,
                  file creation, directory modification, file rename,
                  directory rename, directory creation, etc. are
                  determined by comparing directory snapshots.
        """
    files_renamed = set()
    dirs_renamed = set()
    dirs_modified = set()
    for kev in event_list:
        descriptor = self._descriptors.get_for_fd(kev.ident)
        src_path = descriptor.path
        if is_deleted(kev):
            if descriptor.is_directory:
                self.queue_event(DirDeletedEvent(src_path))
            else:
                self.queue_event(FileDeletedEvent(src_path))
        elif is_attrib_modified(kev):
            if descriptor.is_directory:
                self.queue_event(DirModifiedEvent(src_path))
            else:
                self.queue_event(FileModifiedEvent(src_path))
        elif is_modified(kev):
            if descriptor.is_directory:
                dirs_modified.add(src_path)
            else:
                self.queue_event(FileModifiedEvent(src_path))
        elif is_renamed(kev):
            if descriptor.is_directory:
                dirs_renamed.add(src_path)
            else:
                files_renamed.add(src_path)
    return (files_renamed, dirs_renamed, dirs_modified)