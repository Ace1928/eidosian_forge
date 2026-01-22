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
class ssl:
    CERT_NONE = 0

    class WantReadException(Exception):
        pass

    class WantWriteException(Exception):
        pass

    class SSLContext:
        pass

    class SSLSocket:
        pass

    @classmethod
    def create_default_context(cls, *args, **kwargs):
        raise Exception('no ssl support')