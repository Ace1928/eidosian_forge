import socket
import ssl
import struct
import time
import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import aioquic.quic.events  # type: ignore
import trio
import dns.exception
import dns.inet
from dns._asyncbackend import NullContext
from dns.quic._common import (
class TrioQuicManager(AsyncQuicManager):

    def __init__(self, nursery, conf=None, verify_mode=ssl.CERT_REQUIRED, server_name=None):
        super().__init__(conf, verify_mode, TrioQuicConnection, server_name)
        self._nursery = nursery

    def connect(self, address, port=853, source=None, source_port=0, want_session_ticket=True):
        connection, start = self._connect(address, port, source, source_port, want_session_ticket)
        if start:
            self._nursery.start_soon(connection.run)
        return connection

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        connections = list(self._connections.values())
        for connection in connections:
            await connection.close()
        return False