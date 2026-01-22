import base64
import contextlib
import enum
import errno
import os
import os.path
import selectors
import socket
import struct
import time
from typing import Any, Dict, Optional, Tuple, Union
import dns._features
import dns.exception
import dns.inet
import dns.message
import dns.name
import dns.quic
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.serial
import dns.transaction
import dns.tsig
import dns.xfr
def _udp_recv(sock, max_size, expiration):
    """Reads a datagram from the socket.
    A Timeout exception will be raised if the operation is not completed
    by the expiration time.
    """
    while True:
        try:
            return sock.recvfrom(max_size)
        except BlockingIOError:
            _wait_for_readable(sock, expiration)