import io
import os
import random
import re
import sys
import threading
import time
import zlib
from contextlib import contextmanager
from datetime import datetime
from functools import wraps, partial
import sentry_sdk
from sentry_sdk._compat import text_type, utc_from_timestamp, iteritems
from sentry_sdk.utils import (
from sentry_sdk.envelope import Envelope, Item
from sentry_sdk.tracing import (
from sentry_sdk._types import TYPE_CHECKING
def _flush_loop(self):
    _in_metrics.set(True)
    while self._running or self._force_flush:
        if self._running:
            self._flush_event.wait(self.FLUSHER_SLEEP_TIME)
        self._flush()