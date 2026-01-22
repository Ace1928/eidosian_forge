from __future__ import annotations
import os
import socket
import sys
from contextlib import contextmanager
from itertools import count, cycle
from operator import itemgetter
from typing import TYPE_CHECKING, Any
from kombu import exceptions
from .log import get_logger
from .resource import Resource
from .transport import get_transport_cls, supports_librabbitmq
from .utils.collections import HashedSeq
from .utils.functional import dictfilter, lazy, retry_over_time, shufflecycle
from .utils.objects import cached_property
from .utils.url import as_url, maybe_sanitize_url, parse_url, quote, urlparse
def _do_close_self(self):
    self.declared_entities.clear()
    if self._default_channel:
        self.maybe_close_channel(self._default_channel)
    if self._connection:
        try:
            self.transport.close_connection(self._connection)
        except self.connection_errors + (AttributeError, socket.error):
            pass
        self._connection = None