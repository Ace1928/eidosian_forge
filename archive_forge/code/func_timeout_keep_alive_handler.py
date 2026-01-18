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
def timeout_keep_alive_handler(self) -> None:
    """
        Called on a keep-alive connection if no new data is received after a short
        delay.
        """
    if not self.transport.is_closing():
        event = h11.ConnectionClosed()
        self.conn.send(event)
        self.transport.close()