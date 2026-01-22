import abc
import functools
import json
import os
import signal
import socket
import time
import traceback
import warnings
from contextlib import closing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.distributed.elastic.rendezvous as rdzv
import torch.distributed.elastic.utils.store as store_util
from torch.distributed import Store
from torch.distributed.elastic.events import Event, EventSource, record
from torch.distributed.elastic.metrics import prof, put_metric
from torch.distributed.elastic.multiprocessing import (
from torch.distributed.elastic.utils.logging import get_logger
def _get_socket_with_port() -> socket.socket:
    """Return a free port on localhost.

    The free port is "reserved" by binding a temporary socket on it.
    Close the socket before passing the port to the entity that
    requires it. Usage example::

    sock = _get_socket_with_port()
    with closing(sock):
        port = sock.getsockname()[1]
        sock.close()
        # there is still a race-condition that some other process
        # may grab this port before func() runs
        func(port)
    """
    addrs = socket.getaddrinfo(host='localhost', port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
    for addr in addrs:
        family, type, proto, _, _ = addr
        s = socket.socket(family, type, proto)
        try:
            s.bind(('localhost', 0))
            s.listen(0)
            return s
        except OSError as e:
            s.close()
            log.info('Socket creation attempt failed.', exc_info=e)
    raise RuntimeError('Failed to create a socket')