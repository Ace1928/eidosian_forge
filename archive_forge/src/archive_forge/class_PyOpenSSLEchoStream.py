from __future__ import annotations
import os
import socket as stdlib_socket
import ssl
import sys
import threading
from contextlib import asynccontextmanager, contextmanager, suppress
from functools import partial
from ssl import SSLContext
from typing import (
import pytest
from trio import StapledStream
from trio._tests.pytest_plugin import skip_if_optional_else_raise
from trio.abc import ReceiveStream, SendStream
from trio.testing import (
import trio
from .. import _core, socket as tsocket
from .._abc import Stream
from .._core import BrokenResourceError, ClosedResourceError
from .._core._tests.tutil import slow
from .._highlevel_generic import aclose_forcefully
from .._highlevel_open_tcp_stream import open_tcp_stream
from .._highlevel_socket import SocketListener, SocketStream
from .._ssl import NeedHandshakeError, SSLListener, SSLStream, _is_eof
from .._util import ConflictDetector
from ..testing import (
class PyOpenSSLEchoStream(Stream):

    def __init__(self, sleeper: None=None) -> None:
        ctx = SSL.Context(SSL.SSLv23_METHOD)
        from cryptography.hazmat.bindings.openssl.binding import Binding
        b = Binding()
        if hasattr(b.lib, 'SSL_OP_NO_TLSv1_3'):
            ctx.set_options(b.lib.SSL_OP_NO_TLSv1_3)
        assert not hasattr(SSL, 'OP_NO_TLSv1_4')
        TRIO_TEST_1_CERT.configure_cert(ctx)
        self._conn = SSL.Connection(ctx, None)
        self._conn.set_accept_state()
        self._lot = _core.ParkingLot()
        self._pending_cleartext = bytearray()
        self._send_all_conflict_detector = ConflictDetector('simultaneous calls to PyOpenSSLEchoStream.send_all')
        self._receive_some_conflict_detector = ConflictDetector('simultaneous calls to PyOpenSSLEchoStream.receive_some')
        if sleeper is None:

            async def no_op_sleeper(_: object) -> None:
                return
            self.sleeper = no_op_sleeper
        else:
            self.sleeper = sleeper

    async def aclose(self) -> None:
        self._conn.bio_shutdown()

    def renegotiate_pending(self) -> bool:
        return self._conn.renegotiate_pending()

    def renegotiate(self) -> None:
        assert self._conn.renegotiate()

    async def wait_send_all_might_not_block(self) -> None:
        with self._send_all_conflict_detector:
            await _core.checkpoint()
            await _core.checkpoint()
            await self.sleeper('wait_send_all_might_not_block')

    async def send_all(self, data: bytes) -> None:
        print('  --> transport_stream.send_all')
        with self._send_all_conflict_detector:
            await _core.checkpoint()
            await _core.checkpoint()
            await self.sleeper('send_all')
            self._conn.bio_write(data)
            while True:
                await self.sleeper('send_all')
                try:
                    data = self._conn.recv(1)
                except SSL.ZeroReturnError:
                    self._conn.shutdown()
                    print('renegotiations:', self._conn.total_renegotiations())
                    break
                except SSL.WantReadError:
                    break
                else:
                    self._pending_cleartext += data
            self._lot.unpark_all()
            await self.sleeper('send_all')
            print('  <-- transport_stream.send_all finished')

    async def receive_some(self, nbytes: int | None=None) -> bytes:
        print('  --> transport_stream.receive_some')
        if nbytes is None:
            nbytes = 65536
        with self._receive_some_conflict_detector:
            try:
                await _core.checkpoint()
                await _core.checkpoint()
                while True:
                    await self.sleeper('receive_some')
                    try:
                        return self._conn.bio_read(nbytes)
                    except SSL.WantReadError:
                        if self._pending_cleartext:
                            print('    trying', self._pending_cleartext)
                            try:
                                next_byte = self._pending_cleartext[0:1]
                                self._conn.send(bytes(next_byte))
                            except SSL.WantReadError:
                                try:
                                    return self._conn.bio_read(nbytes)
                                except SSL.WantReadError:
                                    print('parking (a)')
                                    await self._lot.park()
                            else:
                                del self._pending_cleartext[0:1]
                        else:
                            print('parking (b)')
                            await self._lot.park()
            finally:
                await self.sleeper('receive_some')
                print('  <-- transport_stream.receive_some finished')