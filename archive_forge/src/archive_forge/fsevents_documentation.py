from __future__ import with_statement
import sys
import threading
import unicodedata
import _watchdog_fsevents as _fsevents
from wandb_watchdog.events import (
from wandb_watchdog.utils.dirsnapshot import DirectorySnapshot
from wandb_watchdog.observers.api import (

    Mac OS X FSEvents Emitter class.

    :param event_queue:
        The event queue to fill with events.
    :param watch:
        A watch object representing the directory to monitor.
    :type watch:
        :class:`watchdog.observers.api.ObservedWatch`
    :param timeout:
        Read events blocking timeout (in seconds).
    :type timeout:
        ``float``
    