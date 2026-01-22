from __future__ import annotations
import weakref
from typing import TYPE_CHECKING, Callable, Union
import pytest
from .. import _core
from .._sync import *
from .._timeouts import sleep_forever
from ..testing import assert_checkpoints, wait_all_tasks_blocked
from .._channel import open_memory_channel
from .._sync import AsyncContextManagerMixin
class ChannelLock3(AsyncContextManagerMixin):

    def __init__(self) -> None:
        self.s, self.r = open_memory_channel[None](0)
        self.acquired = False

    def acquire_nowait(self) -> None:
        assert not self.acquired
        self.acquired = True

    async def acquire(self) -> None:
        if self.acquired:
            await self.s.send(None)
        else:
            self.acquired = True
            await _core.checkpoint()

    def release(self) -> None:
        try:
            self.r.receive_nowait()
        except _core.WouldBlock:
            assert self.acquired
            self.acquired = False