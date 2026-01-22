import selectors
import socket
import ssl
import struct
import threading
import time
import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import aioquic.quic.events  # type: ignore
import dns.exception
import dns.inet
from dns.quic._common import (
def _add_input(self, data, is_end):
    if self._common_add_input(data, is_end):
        with self._wake_up:
            self._wake_up.notify()