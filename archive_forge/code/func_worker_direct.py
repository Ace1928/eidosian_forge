from __future__ import annotations
import os
import socket
from functools import partial
from kombu.entity import Exchange, Queue
from .functional import memoize
from .text import simple_format
def worker_direct(hostname: str | Queue) -> Queue:
    """Return the :class:`kombu.Queue` being a direct route to a worker.

    Arguments:
        hostname (str, ~kombu.Queue): The fully qualified node name of
            a worker (e.g., ``w1@example.com``).  If passed a
            :class:`kombu.Queue` instance it will simply return
            that instead.
    """
    if isinstance(hostname, Queue):
        return hostname
    return Queue(WORKER_DIRECT_QUEUE_FORMAT.format(hostname=hostname), WORKER_DIRECT_EXCHANGE, hostname)