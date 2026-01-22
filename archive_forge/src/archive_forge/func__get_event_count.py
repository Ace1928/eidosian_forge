import contextlib
import threading
import sys
import warnings
import unittest  # noqa: F401
from traits.api import (
from traits.util.async_trait_wait import wait_for_condition
def _get_event_count(self):
    """ Traits property getter.

        Thread-safe access to event count.

        """
    with self._lock:
        return len(self.events)