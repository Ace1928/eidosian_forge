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
def _do_restore_message(self, payload, exchange, routing_key, pipe, leftmost=False):
    try:
        try:
            payload['headers']['redelivered'] = True
            payload['properties']['delivery_info']['redelivered'] = True
        except KeyError:
            pass
        for queue in self._lookup(exchange, routing_key):
            (pipe.lpush if leftmost else pipe.rpush)(queue, dumps(payload))
    except Exception:
        crit('Could not restore message: %r', payload, exc_info=True)