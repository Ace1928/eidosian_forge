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
def _ensure_indexes(self, database):
    """Ensure indexes on collections."""
    messages = database[self.messages_collection]
    messages.create_index([('queue', 1), ('priority', 1), ('_id', 1)], background=True)
    database[self.broadcast_collection].create_index([('queue', 1)])
    routing = database[self.routing_collection]
    routing.create_index([('queue', 1), ('exchange', 1)])
    if self.ttl:
        messages.create_index([('expire_at', 1)], expireAfterSeconds=0)
        routing.create_index([('expire_at', 1)], expireAfterSeconds=0)
        database[self.queues_collection].create_index([('expire_at', 1)], expireAfterSeconds=0)