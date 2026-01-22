from __future__ import annotations
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from types import TracebackType
from typing import Generic, NamedTuple, TypeVar
from .. import (
from ..abc import Event, ObjectReceiveStream, ObjectSendStream
from ..lowlevel import checkpoint
@dataclass(eq=False)
class MemoryObjectStreamState(Generic[T_Item]):
    max_buffer_size: float = field()
    buffer: deque[T_Item] = field(init=False, default_factory=deque)
    open_send_channels: int = field(init=False, default=0)
    open_receive_channels: int = field(init=False, default=0)
    waiting_receivers: OrderedDict[Event, list[T_Item]] = field(init=False, default_factory=OrderedDict)
    waiting_senders: OrderedDict[Event, T_Item] = field(init=False, default_factory=OrderedDict)

    def statistics(self) -> MemoryObjectStreamStatistics:
        return MemoryObjectStreamStatistics(len(self.buffer), self.max_buffer_size, self.open_send_channels, self.open_receive_channels, len(self.waiting_senders), len(self.waiting_receivers))