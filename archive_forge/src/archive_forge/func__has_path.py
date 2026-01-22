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
def _has_path(self, path):
    """Determines whether a :class:`KeventDescriptor` for the specified
   path exists already in the collection."""
    return path in self._descriptor_for_path