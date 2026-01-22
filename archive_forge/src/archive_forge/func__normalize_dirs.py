from __future__ import annotations
import asyncio
import inspect
import json
import logging
import logging.config
import os
import socket
import ssl
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal
import click
from uvicorn._types import ASGIApplication
from uvicorn.importer import ImportFromStringError, import_from_string
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.middleware.asgi2 import ASGI2Middleware
from uvicorn.middleware.message_logger import MessageLoggerMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from uvicorn.middleware.wsgi import WSGIMiddleware
def _normalize_dirs(dirs: list[str] | str | None) -> list[str]:
    if dirs is None:
        return []
    if isinstance(dirs, str):
        return [dirs]
    return list(set(dirs))