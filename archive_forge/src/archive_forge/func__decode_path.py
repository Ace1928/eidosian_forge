from __future__ import with_statement
import os
import threading
from .inotify_buffer import InotifyBuffer
from wandb_watchdog.observers.api import (
from wandb_watchdog.events import (
from wandb_watchdog.utils import unicode_paths
def _decode_path(self, path):
    """ Decode path only if unicode string was passed to this emitter. """
    if isinstance(self.watch.path, bytes):
        return path
    return unicode_paths.decode(path)