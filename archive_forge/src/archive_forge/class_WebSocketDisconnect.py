from __future__ import annotations
import enum
import json
import typing
from starlette.requests import HTTPConnection
from starlette.types import Message, Receive, Scope, Send
class WebSocketDisconnect(Exception):

    def __init__(self, code: int=1000, reason: typing.Optional[str]=None) -> None:
        self.code = code
        self.reason = reason or ''