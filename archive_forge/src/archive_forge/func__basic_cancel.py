from __future__ import annotations
import functools
import numbers
import socket
from bisect import bisect
from collections import namedtuple
from contextlib import contextmanager
from queue import Empty
from time import time
from vine import promise
from kombu.exceptions import InconsistencyError, VersionMismatch
from kombu.log import get_logger
from kombu.utils.compat import register_after_fork
from kombu.utils.encoding import bytes_to_str
from kombu.utils.eventio import ERR, READ, poll
from kombu.utils.functional import accepts_argument
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from kombu.utils.scheduling import cycle_by_name
from kombu.utils.url import _parse_url
from . import virtual
def _basic_cancel(self, consumer_tag):
    try:
        queue = self._tag_to_queue[consumer_tag]
    except KeyError:
        return
    try:
        self.active_fanout_queues.remove(queue)
    except KeyError:
        pass
    else:
        self._unsubscribe_from(queue)
    try:
        exchange, _ = self._fanout_queues[queue]
        self._fanout_to_queue.pop(exchange)
    except KeyError:
        pass
    ret = super().basic_cancel(consumer_tag)
    self._update_queue_cycle()
    return ret