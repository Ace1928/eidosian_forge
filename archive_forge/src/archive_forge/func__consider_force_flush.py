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
def _consider_force_flush(self):
    total_weight = len(self.buckets) + self._buckets_total_weight
    if total_weight >= self.MAX_WEIGHT:
        self._force_flush = True
        self._flush_event.set()