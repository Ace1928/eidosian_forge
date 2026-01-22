import bisect
import sys
import threading
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from itertools import islice
from operator import itemgetter
from time import time
from typing import Mapping, Optional  # noqa
from weakref import WeakSet, ref
from kombu.clocks import timetuple
from kombu.utils.objects import cached_property
from celery import states
from celery.utils.functional import LRUCache, memoize, pass1
from celery.utils.log import get_logger
def _create_event_handler(self):
    _set = object.__setattr__
    hbmax = self.heartbeat_max
    heartbeats = self.heartbeats
    hb_pop = self.heartbeats.pop
    hb_append = self.heartbeats.append

    def event(type_, timestamp=None, local_received=None, fields=None, max_drift=HEARTBEAT_DRIFT_MAX, abs=abs, int=int, insort=bisect.insort, len=len):
        fields = fields or {}
        for k, v in fields.items():
            _set(self, k, v)
        if type_ == 'offline':
            heartbeats[:] = []
        else:
            if not local_received or not timestamp:
                return
            drift = abs(int(local_received) - int(timestamp))
            if drift > max_drift:
                _warn_drift(self.hostname, drift, local_received, timestamp)
            if local_received:
                hearts = len(heartbeats)
                if hearts > hbmax - 1:
                    hb_pop(0)
                if hearts and local_received > heartbeats[-1]:
                    hb_append(local_received)
                else:
                    insort(heartbeats, local_received)
    return event