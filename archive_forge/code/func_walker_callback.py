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
def walker_callback(path, stat_info, self=self):
    self._register_kevent(path, stat.S_ISDIR(stat_info.st_mode))