import base64
import contextlib
import socket
import struct
import time
from typing import Any, Dict, Optional, Tuple, Union
import dns.asyncbackend
import dns.exception
import dns.inet
import dns.message
import dns.name
import dns.quic
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.transaction
from dns._asyncbackend import NullContext
from dns.query import (
Return the response obtained after sending an asynchronous query via
    DNS-over-QUIC.

    *backend*, a ``dns.asyncbackend.Backend``, or ``None``.  If ``None``,
    the default, then dnspython will use the default backend.

    See :py:func:`dns.query.quic()` for the documentation of the other
    parameters, exceptions, and return type of this method.
    