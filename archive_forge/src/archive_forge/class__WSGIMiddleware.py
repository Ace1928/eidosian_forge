from __future__ import annotations
import asyncio
import concurrent.futures
import io
import sys
import warnings
from collections import deque
from typing import Iterable
from uvicorn._types import (
class _WSGIMiddleware:

    def __init__(self, app: WSGIApp, workers: int=10):
        warnings.warn("Uvicorn's native WSGI implementation is deprecated, you should switch to a2wsgi (`pip install a2wsgi`).", DeprecationWarning)
        self.app = app
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)

    async def __call__(self, scope: HTTPScope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
        assert scope['type'] == 'http'
        instance = WSGIResponder(self.app, self.executor, scope)
        await instance(receive, send)