from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar
from .. import _core, _util
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
class _LockstepSendStream(SendStream):

    def __init__(self, lbq: _LockstepByteQueue):
        self._lbq = lbq

    def close(self) -> None:
        self._lbq.close_sender()

    async def aclose(self) -> None:
        self.close()
        await _core.checkpoint()

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        await self._lbq.send_all(data)

    async def wait_send_all_might_not_block(self) -> None:
        await self._lbq.wait_send_all_might_not_block()