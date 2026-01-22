from __future__ import with_statement
import threading
import os.path
import time
from wandb_watchdog.events import (
from wandb_watchdog.observers.api import (
from wandb_watchdog.observers.winapi import (
class WindowsApiObserver(BaseObserver):
    """
    Observer thread that schedules watching directories and dispatches
    calls to event handlers.
    """

    def __init__(self, timeout=DEFAULT_OBSERVER_TIMEOUT):
        BaseObserver.__init__(self, emitter_class=WindowsApiEmitter, timeout=timeout)