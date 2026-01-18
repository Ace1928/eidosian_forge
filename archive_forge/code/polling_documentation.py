from __future__ import with_statement
import os
import threading
from functools import partial
from wandb_watchdog.utils import stat as default_stat
from wandb_watchdog.utils.dirsnapshot import DirectorySnapshot, DirectorySnapshotDiff
from wandb_watchdog.observers.api import (
from wandb_watchdog.events import (

        :param stat: stat function. See ``os.stat`` for details.
        :param listdir: listdir function. See ``os.listdir`` for details.
        :type polling_interval: float
        :param polling_interval: interval in seconds between polling the file system.
        