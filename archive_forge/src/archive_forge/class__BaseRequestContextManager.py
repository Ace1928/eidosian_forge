import asyncio
import base64
import hashlib
import json
import os
import sys
import traceback
import warnings
from contextlib import suppress
from types import SimpleNamespace, TracebackType
from typing import (
import attr
from multidict import CIMultiDict, MultiDict, MultiDictProxy, istr
from yarl import URL
from . import hdrs, http, payload
from .abc import AbstractCookieJar
from .client_exceptions import (
from .client_reqrep import (
from .client_ws import ClientWebSocketResponse as ClientWebSocketResponse
from .connector import (
from .cookiejar import CookieJar
from .helpers import (
from .http import WS_KEY, HttpVersion, WebSocketReader, WebSocketWriter
from .http_websocket import WSHandshakeError, WSMessage, ws_ext_gen, ws_ext_parse
from .streams import FlowControlDataQueue
from .tracing import Trace, TraceConfig
from .typedefs import JSONEncoder, LooseCookies, LooseHeaders, StrOrURL
class _BaseRequestContextManager(Coroutine[Any, Any, _RetType], Generic[_RetType]):
    __slots__ = ('_coro', '_resp')

    def __init__(self, coro: Coroutine['asyncio.Future[Any]', None, _RetType]) -> None:
        self._coro = coro

    def send(self, arg: None) -> 'asyncio.Future[Any]':
        return self._coro.send(arg)

    def throw(self, *args: Any, **kwargs: Any) -> 'asyncio.Future[Any]':
        return self._coro.throw(*args, **kwargs)

    def close(self) -> None:
        return self._coro.close()

    def __await__(self) -> Generator[Any, None, _RetType]:
        ret = self._coro.__await__()
        return ret

    def __iter__(self) -> Generator[Any, None, _RetType]:
        return self.__await__()

    async def __aenter__(self) -> _RetType:
        self._resp = await self._coro
        return self._resp