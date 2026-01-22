from __future__ import annotations
import asyncio
import http
import logging
from typing import Any, Callable, Literal, cast
from urllib.parse import unquote
import h11
from h11._connection import DEFAULT_MAX_INCOMPLETE_EVENT_SIZE
from uvicorn._types import (
from uvicorn.config import Config
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.protocols.http.flow_control import (
from uvicorn.protocols.utils import (
from uvicorn.server import ServerState
def _get_upgrade(self) -> bytes | None:
    connection = []
    upgrade = None
    for name, value in self.headers:
        if name == b'connection':
            connection = [token.lower().strip() for token in value.split(b',')]
        if name == b'upgrade':
            upgrade = value.lower()
    if b'upgrade' in connection:
        return upgrade
    return None