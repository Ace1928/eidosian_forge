from __future__ import annotations
import sys
import types
from typing import (
class WebSocketAcceptEvent(TypedDict):
    type: Literal['websocket.accept']
    subprotocol: NotRequired[str | None]
    headers: NotRequired[Iterable[tuple[bytes, bytes]]]