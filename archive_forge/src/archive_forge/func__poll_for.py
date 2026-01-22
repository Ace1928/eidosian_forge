from __future__ import generators
import errno
import select
import socket
import struct
import sys
import time
import dns.exception
import dns.inet
import dns.name
import dns.message
import dns.rcode
import dns.rdataclass
import dns.rdatatype
from ._compat import long, string_types, PY3
def _poll_for(fd, readable, writable, error, timeout):
    """Poll polling backend."""
    event_mask = 0
    if readable:
        event_mask |= select.POLLIN
    if writable:
        event_mask |= select.POLLOUT
    if error:
        event_mask |= select.POLLERR
    pollable = select.poll()
    pollable.register(fd, event_mask)
    if timeout:
        event_list = pollable.poll(long(timeout * 1000))
    else:
        event_list = pollable.poll()
    return bool(event_list)