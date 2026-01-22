import logging
import os
import string
import sys
import time
from taskflow import engines
from taskflow.engines.worker_based import worker
from taskflow.patterns import linear_flow as lf
from taskflow import task
from taskflow.types import notifier
from taskflow.utils import threading_utils
class EventReporter(task.Task):
    """This is the task that will be running 'remotely' (not really remote)."""
    EVENTS = tuple(string.ascii_uppercase)
    EVENT_DELAY = 0.1

    def execute(self):
        for i, e in enumerate(self.EVENTS):
            details = {'leftover': self.EVENTS[i:]}
            self.notifier.notify(e, details)
            time.sleep(self.EVENT_DELAY)