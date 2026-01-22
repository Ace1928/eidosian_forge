import copy
import functools
import socket
import struct
import time
from typing import Any, Optional
import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import dns.inet
class AsyncQuicManager(BaseQuicManager):

    def connect(self, address, port=853, source=None, source_port=0):
        raise NotImplementedError