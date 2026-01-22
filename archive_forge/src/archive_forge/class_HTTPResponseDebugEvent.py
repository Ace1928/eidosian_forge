from __future__ import annotations
import sys
import types
from typing import (
class HTTPResponseDebugEvent(TypedDict):
    type: Literal['http.response.debug']
    info: dict[str, object]