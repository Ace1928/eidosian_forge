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
def _key_prefix(self, queue):
    """Create and return the `queue` with the proper prefix.

        Arguments:
        ---------
            queue (str): The name of the queue.
        """
    return f'{self.prefix}/{queue}'