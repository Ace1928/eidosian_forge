import os
import logging
import unicodedata
from threading import Thread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.events import (
from wandb_watchdog.observers.api import (
import AppKit
from FSEvents import (
from FSEvents import (
class FSEventsObserver2(BaseObserver):

    def __init__(self, timeout=DEFAULT_OBSERVER_TIMEOUT):
        BaseObserver.__init__(self, emitter_class=FSEventsEmitter, timeout=timeout)