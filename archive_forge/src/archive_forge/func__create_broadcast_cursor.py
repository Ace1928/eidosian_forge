from __future__ import annotations
import datetime
from queue import Empty
import pymongo
from pymongo import MongoClient, errors, uri_parser
from pymongo.cursor import CursorType
from kombu.exceptions import VersionMismatch
from kombu.utils.compat import _detect_environment
from kombu.utils.encoding import bytes_to_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from kombu.utils.url import maybe_sanitize_url
from . import virtual
from .base import to_rabbitmq_queue_arguments
def _create_broadcast_cursor(self, exchange, routing_key, pattern, queue):
    if pymongo.version_tuple >= (3,):
        query = {'filter': {'queue': exchange}, 'cursor_type': CursorType.TAILABLE}
    else:
        query = {'query': {'queue': exchange}, 'tailable': True}
    cursor = self.broadcast.find(**query)
    ret = self._broadcast_cursors[queue] = BroadcastCursor(cursor)
    return ret