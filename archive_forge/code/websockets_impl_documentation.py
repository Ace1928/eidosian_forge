from __future__ import annotations
import asyncio
import http
import logging
from typing import Any, Literal, Optional, Sequence, cast
from urllib.parse import unquote
import websockets
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.extensions.permessage_deflate import ServerPerMessageDeflateFactory
from websockets.legacy.server import HTTPResponse
from websockets.server import WebSocketServerProtocol
from websockets.typing import Subprotocol
from uvicorn._types import (
from uvicorn.config import Config
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.protocols.utils import (
from uvicorn.server import ServerState

        Wrapper around the ASGI callable, handling exceptions and unexpected
        termination states.
        