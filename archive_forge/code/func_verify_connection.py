from __future__ import annotations
import os
import socket
from collections import defaultdict
from contextlib import contextmanager
from queue import Empty
from kombu.exceptions import ChannelError
from kombu.log import get_logger
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
def verify_connection(self, connection):
    """Verify the connection works."""
    port = connection.client.port or self.default_port
    host = connection.client.hostname or DEFAULT_HOST
    logger.debug('Verify Etcd connection to %s:%s', host, port)
    try:
        etcd.Client(host=host, port=int(port))
        return True
    except ValueError:
        pass
    return False