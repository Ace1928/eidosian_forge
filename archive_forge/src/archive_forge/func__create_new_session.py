from __future__ import annotations
import socket
import uuid
from collections import defaultdict
from contextlib import contextmanager
from queue import Empty
from time import monotonic
from kombu.exceptions import ChannelError
from kombu.log import get_logger
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
def _create_new_session(self):
    logger.debug('Creating session %s with TTL %s', self.lock_name, self.session_ttl)
    session_id = self.client.session.create(name=self.lock_name, ttl=self.session_ttl)
    logger.debug('Created session %s with id %s', self.lock_name, session_id)
    return session_id