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
class BroadcastCursor:
    """Cursor for broadcast queues."""

    def __init__(self, cursor):
        self._cursor = cursor
        self._offset = 0
        self.purge(rewind=False)

    def get_size(self):
        return self._cursor.collection.count_documents({}) - self._offset

    def close(self):
        self._cursor.close()

    def purge(self, rewind=True):
        if rewind:
            self._cursor.rewind()
        self._offset = self._cursor.collection.count_documents({})
        self._cursor = self._cursor.skip(self._offset)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                msg = next(self._cursor)
            except pymongo.errors.OperationFailure as exc:
                if 'not valid at server' in str(exc):
                    self.purge()
                    continue
                raise
            else:
                break
        self._offset += 1
        return msg
    next = __next__