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
class _WSRequestContextManager(_BaseRequestContextManager[ClientWebSocketResponse]):
    __slots__ = ()

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]) -> None:
        await self._resp.close()