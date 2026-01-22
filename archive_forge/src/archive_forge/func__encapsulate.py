import copy
import functools
import socket
import struct
import time
from typing import Any, Optional
import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import dns.inet
def _encapsulate(self, datagram):
    l = len(datagram)
    return struct.pack('!H', l) + datagram