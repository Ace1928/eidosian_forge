from __future__ import annotations
import sys
import types
from typing import (
class HTTPRequestEvent(TypedDict):
    type: Literal['http.request']
    body: bytes
    more_body: bool