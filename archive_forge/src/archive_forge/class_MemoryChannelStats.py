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
@attrs.frozen
class MemoryChannelStats:
    current_buffer_used: int
    max_buffer_size: int | float
    open_send_channels: int
    open_receive_channels: int
    tasks_waiting_send: int
    tasks_waiting_receive: int