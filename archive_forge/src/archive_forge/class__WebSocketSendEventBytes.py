from __future__ import annotations
import sys
import types
from typing import (
class _WebSocketSendEventBytes(TypedDict):
    type: Literal['websocket.send']
    bytes: bytes
    text: NotRequired[None]