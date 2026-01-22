import socket
import sys
import threading
from contextlib import suppress
from . import Adapter
from .. import errors
from .._compat import IS_ABOVE_OPENSSL10
from ..makefile import StreamReader, StreamWriter
from ..server import HTTPServer
def _sni_callback(sock, sni, context):
    """Handle the SNI callback to tag the socket with the SNI."""
    sock.sni = sni