from __future__ import annotations
import threading
from json import dumps, loads
from queue import Empty
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from kombu.transport import virtual
from kombu.utils import cached_property
from kombu.utils.encoding import bytes_to_str
from .models import Message as MessageBase
from .models import ModelBase
from .models import Queue as QueueBase
from .models import class_registry, metadata
def _configure_entity_tablenames(self, opts):
    self.queue_tablename = opts.get('queue_tablename', 'kombu_queue')
    self.message_tablename = opts.get('message_tablename', 'kombu_message')
    self.queue_cls and self.message_cls