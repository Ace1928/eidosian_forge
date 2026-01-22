import copy
import functools
import socket
import struct
import time
from typing import Any, Optional
import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import dns.inet
def _get_timer_values(self, closed_is_special=True):
    now = time.time()
    expiration = self._connection.get_timer()
    if expiration is None:
        expiration = now + 3600
    interval = max(expiration - now, 0)
    if self._closed and closed_is_special:
        interval = min(interval, 0.05)
    return (expiration, interval)