from __future__ import annotations
from collections import OrderedDict, deque
from math import inf
from typing import (
import attrs
from outcome import Error, Value
import trio
from ._abc import ReceiveChannel, ReceiveType, SendChannel, SendType, T
from ._core import Abort, RaiseCancelT, Task, enable_ki_protection
from ._util import NoPublicConstructor, final, generic_function
@attrs.define
class MemoryChannelState(Generic[T]):
    max_buffer_size: int | float
    data: deque[T] = attrs.Factory(deque)
    open_send_channels: int = 0
    open_receive_channels: int = 0
    send_tasks: OrderedDict[Task, T] = attrs.Factory(OrderedDict)
    receive_tasks: OrderedDict[Task, None] = attrs.Factory(OrderedDict)

    def statistics(self) -> MemoryChannelStats:
        return MemoryChannelStats(current_buffer_used=len(self.data), max_buffer_size=self.max_buffer_size, open_send_channels=self.open_send_channels, open_receive_channels=self.open_receive_channels, tasks_waiting_send=len(self.send_tasks), tasks_waiting_receive=len(self.receive_tasks))