from __future__ import annotations
import sys
import types
from typing import (
class HTTPResponseTrailersEvent(TypedDict):
    type: Literal['http.response.trailers']
    headers: Iterable[tuple[bytes, bytes]]
    more_trailers: bool