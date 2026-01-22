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
def _add_pending_task_child(self, task):
    try:
        ch = self._tasks_to_resolve[task.parent_id]
    except KeyError:
        ch = self._tasks_to_resolve[task.parent_id] = WeakSet()
    ch.add(task)