from __future__ import with_statement
import os
import threading
from functools import partial
from wandb_watchdog.utils import stat as default_stat
from wandb_watchdog.utils.dirsnapshot import DirectorySnapshot, DirectorySnapshotDiff
from wandb_watchdog.observers.api import (
from wandb_watchdog.events import (
class PollingObserver(BaseObserver):
    """
    Platform-independent observer that polls a directory to detect file
    system changes.
    """

    def __init__(self, timeout=DEFAULT_OBSERVER_TIMEOUT):
        BaseObserver.__init__(self, emitter_class=PollingEmitter, timeout=timeout)