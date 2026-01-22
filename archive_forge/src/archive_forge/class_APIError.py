from __future__ import annotations
from typing import Any, Optional, cast
from typing_extensions import Literal
import httpx
from ._utils import is_dict
class APIError(OpenAIError):
    message: str
    request: httpx.Request
    body: object | None
    "The API response body.\n\n    If the API responded with a valid JSON structure then this property will be the\n    decoded result.\n\n    If it isn't a valid JSON structure then this will be the raw response.\n\n    If there was no response associated with this error then it will be `None`.\n    "
    code: Optional[str] = None
    param: Optional[str] = None
    type: Optional[str]

    def __init__(self, message: str, request: httpx.Request, *, body: object | None) -> None:
        super().__init__(message)
        self.request = request
        self.message = message
        self.body = body
        if is_dict(body):
            self.code = cast(Any, body.get('code'))
            self.param = cast(Any, body.get('param'))
            self.type = cast(Any, body.get('type'))
        else:
            self.code = None
            self.param = None
            self.type = None