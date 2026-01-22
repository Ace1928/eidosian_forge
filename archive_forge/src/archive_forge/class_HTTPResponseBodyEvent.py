from __future__ import annotations
import sys
import types
from typing import (
class HTTPResponseBodyEvent(TypedDict):
    type: Literal['http.response.body']
    body: bytes
    more_body: NotRequired[bool]