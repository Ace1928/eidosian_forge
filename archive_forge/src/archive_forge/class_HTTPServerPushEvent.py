from __future__ import annotations
import sys
import types
from typing import (
class HTTPServerPushEvent(TypedDict):
    type: Literal['http.response.push']
    path: str
    headers: Iterable[tuple[bytes, bytes]]