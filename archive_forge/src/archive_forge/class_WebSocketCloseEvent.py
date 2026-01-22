from __future__ import annotations
import sys
import types
from typing import (
class WebSocketCloseEvent(TypedDict):
    type: Literal['websocket.close']
    code: NotRequired[int]
    reason: NotRequired[str | None]