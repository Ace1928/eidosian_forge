from __future__ import annotations
import enum
import json
import typing
from starlette.requests import HTTPConnection
from starlette.types import Message, Receive, Scope, Send

        Send ASGI websocket messages, ensuring valid state transitions.
        