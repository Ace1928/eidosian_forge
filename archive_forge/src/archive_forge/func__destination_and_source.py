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
def _destination_and_source(af, where, port, source, source_port):
    if af is None:
        try:
            af = dns.inet.af_for_address(where)
        except Exception:
            af = dns.inet.AF_INET
    if af == dns.inet.AF_INET:
        destination = (where, port)
        if source is not None or source_port != 0:
            if source is None:
                source = '0.0.0.0'
            source = (source, source_port)
    elif af == dns.inet.AF_INET6:
        destination = (where, port, 0, 0)
        if source is not None or source_port != 0:
            if source is None:
                source = '::'
            source = (source, source_port, 0, 0)
    return (af, destination, source)