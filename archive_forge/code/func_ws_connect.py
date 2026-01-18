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
def ws_connect(self, url: StrOrURL, *, method: str=hdrs.METH_GET, protocols: Iterable[str]=(), timeout: float=10.0, receive_timeout: Optional[float]=None, autoclose: bool=True, autoping: bool=True, heartbeat: Optional[float]=None, auth: Optional[BasicAuth]=None, origin: Optional[str]=None, params: Optional[Mapping[str, str]]=None, headers: Optional[LooseHeaders]=None, proxy: Optional[StrOrURL]=None, proxy_auth: Optional[BasicAuth]=None, ssl: Union[SSLContext, bool, None, Fingerprint]=True, verify_ssl: Optional[bool]=None, fingerprint: Optional[bytes]=None, ssl_context: Optional[SSLContext]=None, proxy_headers: Optional[LooseHeaders]=None, compress: int=0, max_msg_size: int=4 * 1024 * 1024) -> '_WSRequestContextManager':
    """Initiate websocket connection."""
    return _WSRequestContextManager(self._ws_connect(url, method=method, protocols=protocols, timeout=timeout, receive_timeout=receive_timeout, autoclose=autoclose, autoping=autoping, heartbeat=heartbeat, auth=auth, origin=origin, params=params, headers=headers, proxy=proxy, proxy_auth=proxy_auth, ssl=ssl, verify_ssl=verify_ssl, fingerprint=fingerprint, ssl_context=ssl_context, proxy_headers=proxy_headers, compress=compress, max_msg_size=max_msg_size))