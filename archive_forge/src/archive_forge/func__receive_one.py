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
def _receive_one(self, c):
    response = None
    try:
        response = c.parse_response()
    except self.connection_errors:
        self._in_listen = None
        raise
    if isinstance(response, (list, tuple)):
        payload = self._handle_message(c, response)
        if bytes_to_str(payload['type']).endswith('message'):
            channel = bytes_to_str(payload['channel'])
            if payload['data']:
                if channel[0] == '/':
                    _, _, channel = channel.partition('.')
                try:
                    message = loads(bytes_to_str(payload['data']))
                except (TypeError, ValueError):
                    warn('Cannot process event on channel %r: %s', channel, repr(payload)[:4096], exc_info=1)
                    raise Empty()
                exchange = channel.split('/', 1)[0]
                self.connection._deliver(message, self._fanout_to_queue[exchange])
                return True