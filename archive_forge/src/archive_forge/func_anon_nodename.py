from __future__ import annotations
import os
import socket
from functools import partial
from kombu.entity import Exchange, Queue
from .functional import memoize
from .text import simple_format
def anon_nodename(hostname: str | None=None, prefix: str='gen') -> str:
    """Return the nodename for this process (not a worker).

    This is used for e.g. the origin task message field.
    """
    return nodename(''.join([prefix, str(os.getpid())]), hostname or gethostname())