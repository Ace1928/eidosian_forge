import copy
import functools
import socket
import struct
import time
from typing import Any, Optional
import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import dns.inet
def _timeout_from_expiration(self, expiration):
    if expiration is not None:
        timeout = max(expiration - time.time(), 0.0)
    else:
        timeout = None
    return timeout