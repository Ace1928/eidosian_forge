from __future__ import annotations
import logging # isort:skip
from typing import Any, Awaitable, Callable
from tornado import locks
from tornado.websocket import WebSocketClientConnection
 Read a message from websocket and execute a callback.

        