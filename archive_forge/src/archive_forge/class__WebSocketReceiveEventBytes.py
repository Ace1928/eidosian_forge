from __future__ import annotations
import sys
import types
from typing import (
class _WebSocketReceiveEventBytes(TypedDict):
    type: Literal['websocket.receive']
    bytes: bytes
    text: NotRequired[None]