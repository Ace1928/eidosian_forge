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
class _SessionRequestContextManager:
    __slots__ = ('_coro', '_resp', '_session')

    def __init__(self, coro: Coroutine['asyncio.Future[Any]', None, ClientResponse], session: ClientSession) -> None:
        self._coro = coro
        self._resp: Optional[ClientResponse] = None
        self._session = session

    async def __aenter__(self) -> ClientResponse:
        try:
            self._resp = await self._coro
        except BaseException:
            await self._session.close()
            raise
        else:
            return self._resp

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]) -> None:
        assert self._resp is not None
        self._resp.close()
        await self._session.close()