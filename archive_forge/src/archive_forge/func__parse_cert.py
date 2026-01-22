import socket
import sys
import threading
from contextlib import suppress
from . import Adapter
from .. import errors
from .._compat import IS_ABOVE_OPENSSL10
from ..makefile import StreamReader, StreamWriter
from ..server import HTTPServer
def _parse_cert(certificate, private_key, certificate_chain):
    """Parse a certificate."""
    with suppress(AttributeError, ssl.SSLError, OSError):
        return _loopback_for_cert(certificate, private_key, certificate_chain)
    with suppress(Exception):
        return ssl._ssl._test_decode_cert(certificate)
    return {}