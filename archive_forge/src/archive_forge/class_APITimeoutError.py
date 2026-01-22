from __future__ import annotations
from typing import Any, Optional, cast
from typing_extensions import Literal
import httpx
from ._utils import is_dict
class APITimeoutError(APIConnectionError):

    def __init__(self, request: httpx.Request) -> None:
        super().__init__(message='Request timed out.', request=request)