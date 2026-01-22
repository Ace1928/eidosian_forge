from __future__ import annotations
import struct
from typing import Awaitable, overload
import zmq
import zmq.asyncio
from zmq._typing import TypedDict
from zmq.error import _check_version
class _MonitorMessage(TypedDict):
    event: int
    value: int
    endpoint: bytes