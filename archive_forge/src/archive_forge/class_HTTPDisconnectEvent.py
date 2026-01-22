from __future__ import annotations
import sys
import types
from typing import (
class HTTPDisconnectEvent(TypedDict):
    type: Literal['http.disconnect']