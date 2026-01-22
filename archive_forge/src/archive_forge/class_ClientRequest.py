import asyncio
import codecs
import contextlib
import functools
import io
import re
import sys
import traceback
import warnings
from hashlib import md5, sha1, sha256
from http.cookies import CookieError, Morsel, SimpleCookie
from types import MappingProxyType, TracebackType
from typing import (
import attr
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL
from . import hdrs, helpers, http, multipart, payload
from .abc import AbstractStreamWriter
from .client_exceptions import (
from .compression_utils import HAS_BROTLI
from .formdata import FormData
from .helpers import (
from .http import (
from .log import client_logger
from .streams import StreamReader
from .typedefs import (
class ClientRequest:
    GET_METHODS = {hdrs.METH_GET, hdrs.METH_HEAD, hdrs.METH_OPTIONS, hdrs.METH_TRACE}
    POST_METHODS = {hdrs.METH_PATCH, hdrs.METH_POST, hdrs.METH_PUT}
    ALL_METHODS = GET_METHODS.union(POST_METHODS).union({hdrs.METH_DELETE})
    DEFAULT_HEADERS = {hdrs.ACCEPT: '*/*', hdrs.ACCEPT_ENCODING: _gen_default_accept_encoding()}
    body = b''
    auth = None
    response = None
    __writer = None
    _continue = None

    def __init__(self, method: str, url: URL, *, params: Optional[Mapping[str, str]]=None, headers: Optional[LooseHeaders]=None, skip_auto_headers: Iterable[str]=frozenset(), data: Any=None, cookies: Optional[LooseCookies]=None, auth: Optional[BasicAuth]=None, version: http.HttpVersion=http.HttpVersion11, compress: Optional[str]=None, chunked: Optional[bool]=None, expect100: bool=False, loop: Optional[asyncio.AbstractEventLoop]=None, response_class: Optional[Type['ClientResponse']]=None, proxy: Optional[URL]=None, proxy_auth: Optional[BasicAuth]=None, timer: Optional[BaseTimerContext]=None, session: Optional['ClientSession']=None, ssl: Union[SSLContext, bool, Fingerprint]=True, proxy_headers: Optional[LooseHeaders]=None, traces: Optional[List['Trace']]=None, trust_env: bool=False, server_hostname: Optional[str]=None):
        if loop is None:
            loop = asyncio.get_event_loop()
        match = _CONTAINS_CONTROL_CHAR_RE.search(method)
        if match:
            raise ValueError(f'Method cannot contain non-token characters {method!r} (found at least {{match.group()!r}})')
        assert isinstance(url, URL), url
        assert isinstance(proxy, (URL, type(None))), proxy
        self._session = cast('ClientSession', session)
        if params:
            q = MultiDict(url.query)
            url2 = url.with_query(params)
            q.extend(url2.query)
            url = url.with_query(q)
        self.original_url = url
        self.url = url.with_fragment(None)
        self.method = method.upper()
        self.chunked = chunked
        self.compress = compress
        self.loop = loop
        self.length = None
        if response_class is None:
            real_response_class = ClientResponse
        else:
            real_response_class = response_class
        self.response_class: Type[ClientResponse] = real_response_class
        self._timer = timer if timer is not None else TimerNoop()
        self._ssl = ssl if ssl is not None else True
        self.server_hostname = server_hostname
        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(1))
        self.update_version(version)
        self.update_host(url)
        self.update_headers(headers)
        self.update_auto_headers(skip_auto_headers)
        self.update_cookies(cookies)
        self.update_content_encoding(data)
        self.update_auth(auth, trust_env)
        self.update_proxy(proxy, proxy_auth, proxy_headers)
        self.update_body_from_data(data)
        if data is not None or self.method not in self.GET_METHODS:
            self.update_transfer_encoding()
        self.update_expect_continue(expect100)
        if traces is None:
            traces = []
        self._traces = traces

    def __reset_writer(self, _: object=None) -> None:
        self.__writer = None

    @property
    def _writer(self) -> Optional['asyncio.Task[None]']:
        return self.__writer

    @_writer.setter
    def _writer(self, writer: Optional['asyncio.Task[None]']) -> None:
        if self.__writer is not None:
            self.__writer.remove_done_callback(self.__reset_writer)
        self.__writer = writer
        if writer is not None:
            writer.add_done_callback(self.__reset_writer)

    def is_ssl(self) -> bool:
        return self.url.scheme in ('https', 'wss')

    @property
    def ssl(self) -> Union['SSLContext', bool, Fingerprint]:
        return self._ssl

    @property
    def connection_key(self) -> ConnectionKey:
        proxy_headers = self.proxy_headers
        if proxy_headers:
            h: Optional[int] = hash(tuple(((k, v) for k, v in proxy_headers.items())))
        else:
            h = None
        return ConnectionKey(self.host, self.port, self.is_ssl(), self.ssl, self.proxy, self.proxy_auth, h)

    @property
    def host(self) -> str:
        ret = self.url.raw_host
        assert ret is not None
        return ret

    @property
    def port(self) -> Optional[int]:
        return self.url.port

    @property
    def request_info(self) -> RequestInfo:
        headers: CIMultiDictProxy[str] = CIMultiDictProxy(self.headers)
        return RequestInfo(self.url, self.method, headers, self.original_url)

    def update_host(self, url: URL) -> None:
        """Update destination host, port and connection type (ssl)."""
        if not url.raw_host:
            raise InvalidURL(url)
        username, password = (url.user, url.password)
        if username:
            self.auth = helpers.BasicAuth(username, password or '')

    def update_version(self, version: Union[http.HttpVersion, str]) -> None:
        """Convert request version to two elements tuple.

        parser HTTP version '1.1' => (1, 1)
        """
        if isinstance(version, str):
            v = [part.strip() for part in version.split('.', 1)]
            try:
                version = http.HttpVersion(int(v[0]), int(v[1]))
            except ValueError:
                raise ValueError(f'Can not parse http version number: {version}') from None
        self.version = version

    def update_headers(self, headers: Optional[LooseHeaders]) -> None:
        """Update request headers."""
        self.headers: CIMultiDict[str] = CIMultiDict()
        netloc = cast(str, self.url.raw_host)
        if helpers.is_ipv6_address(netloc):
            netloc = f'[{netloc}]'
        netloc = netloc.rstrip('.')
        if self.url.port is not None and (not self.url.is_default_port()):
            netloc += ':' + str(self.url.port)
        self.headers[hdrs.HOST] = netloc
        if headers:
            if isinstance(headers, (dict, MultiDictProxy, MultiDict)):
                headers = headers.items()
            for key, value in headers:
                if key.lower() == 'host':
                    self.headers[key] = value
                else:
                    self.headers.add(key, value)

    def update_auto_headers(self, skip_auto_headers: Iterable[str]) -> None:
        self.skip_auto_headers = CIMultiDict(((hdr, None) for hdr in sorted(skip_auto_headers)))
        used_headers = self.headers.copy()
        used_headers.extend(self.skip_auto_headers)
        for hdr, val in self.DEFAULT_HEADERS.items():
            if hdr not in used_headers:
                self.headers.add(hdr, val)
        if hdrs.USER_AGENT not in used_headers:
            self.headers[hdrs.USER_AGENT] = SERVER_SOFTWARE

    def update_cookies(self, cookies: Optional[LooseCookies]) -> None:
        """Update request cookies header."""
        if not cookies:
            return
        c = SimpleCookie()
        if hdrs.COOKIE in self.headers:
            c.load(self.headers.get(hdrs.COOKIE, ''))
            del self.headers[hdrs.COOKIE]
        if isinstance(cookies, Mapping):
            iter_cookies = cookies.items()
        else:
            iter_cookies = cookies
        for name, value in iter_cookies:
            if isinstance(value, Morsel):
                mrsl_val = value.get(value.key, Morsel())
                mrsl_val.set(value.key, value.value, value.coded_value)
                c[name] = mrsl_val
            else:
                c[name] = value
        self.headers[hdrs.COOKIE] = c.output(header='', sep=';').strip()

    def update_content_encoding(self, data: Any) -> None:
        """Set request content encoding."""
        if data is None:
            return
        enc = self.headers.get(hdrs.CONTENT_ENCODING, '').lower()
        if enc:
            if self.compress:
                raise ValueError('compress can not be set if Content-Encoding header is set')
        elif self.compress:
            if not isinstance(self.compress, str):
                self.compress = 'deflate'
            self.headers[hdrs.CONTENT_ENCODING] = self.compress
            self.chunked = True

    def update_transfer_encoding(self) -> None:
        """Analyze transfer-encoding header."""
        te = self.headers.get(hdrs.TRANSFER_ENCODING, '').lower()
        if 'chunked' in te:
            if self.chunked:
                raise ValueError('chunked can not be set if "Transfer-Encoding: chunked" header is set')
        elif self.chunked:
            if hdrs.CONTENT_LENGTH in self.headers:
                raise ValueError('chunked can not be set if Content-Length header is set')
            self.headers[hdrs.TRANSFER_ENCODING] = 'chunked'
        elif hdrs.CONTENT_LENGTH not in self.headers:
            self.headers[hdrs.CONTENT_LENGTH] = str(len(self.body))

    def update_auth(self, auth: Optional[BasicAuth], trust_env: bool=False) -> None:
        """Set basic auth."""
        if auth is None:
            auth = self.auth
        if auth is None and trust_env and (self.url.host is not None):
            netrc_obj = netrc_from_env()
            with contextlib.suppress(LookupError):
                auth = basicauth_from_netrc(netrc_obj, self.url.host)
        if auth is None:
            return
        if not isinstance(auth, helpers.BasicAuth):
            raise TypeError('BasicAuth() tuple is required instead')
        self.headers[hdrs.AUTHORIZATION] = auth.encode()

    def update_body_from_data(self, body: Any) -> None:
        if body is None:
            return
        if isinstance(body, FormData):
            body = body()
        try:
            body = payload.PAYLOAD_REGISTRY.get(body, disposition=None)
        except payload.LookupError:
            body = FormData(body)()
        self.body = body
        if not self.chunked:
            if hdrs.CONTENT_LENGTH not in self.headers:
                size = body.size
                if size is None:
                    self.chunked = True
                elif hdrs.CONTENT_LENGTH not in self.headers:
                    self.headers[hdrs.CONTENT_LENGTH] = str(size)
        assert body.headers
        for key, value in body.headers.items():
            if key in self.headers:
                continue
            if key in self.skip_auto_headers:
                continue
            self.headers[key] = value

    def update_expect_continue(self, expect: bool=False) -> None:
        if expect:
            self.headers[hdrs.EXPECT] = '100-continue'
        elif self.headers.get(hdrs.EXPECT, '').lower() == '100-continue':
            expect = True
        if expect:
            self._continue = self.loop.create_future()

    def update_proxy(self, proxy: Optional[URL], proxy_auth: Optional[BasicAuth], proxy_headers: Optional[LooseHeaders]) -> None:
        if proxy_auth and (not isinstance(proxy_auth, helpers.BasicAuth)):
            raise ValueError('proxy_auth must be None or BasicAuth() tuple')
        self.proxy = proxy
        self.proxy_auth = proxy_auth
        self.proxy_headers = proxy_headers

    def keep_alive(self) -> bool:
        if self.version < HttpVersion10:
            return False
        if self.version == HttpVersion10:
            if self.headers.get(hdrs.CONNECTION) == 'keep-alive':
                return True
            else:
                return False
        elif self.headers.get(hdrs.CONNECTION) == 'close':
            return False
        return True

    async def write_bytes(self, writer: AbstractStreamWriter, conn: 'Connection') -> None:
        """Support coroutines that yields bytes objects."""
        if self._continue is not None:
            try:
                await writer.drain()
                await self._continue
            except asyncio.CancelledError:
                return
        protocol = conn.protocol
        assert protocol is not None
        try:
            if isinstance(self.body, payload.Payload):
                await self.body.write(writer)
            else:
                if isinstance(self.body, (bytes, bytearray)):
                    self.body = (self.body,)
                for chunk in self.body:
                    await writer.write(chunk)
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                protocol.set_exception(exc)
            else:
                new_exc = ClientOSError(exc.errno, 'Can not write request body for %s' % self.url)
                new_exc.__context__ = exc
                new_exc.__cause__ = exc
                protocol.set_exception(new_exc)
        except asyncio.CancelledError:
            await writer.write_eof()
        except Exception as exc:
            protocol.set_exception(exc)
        else:
            await writer.write_eof()
            protocol.start_timeout()

    async def send(self, conn: 'Connection') -> 'ClientResponse':
        if self.method == hdrs.METH_CONNECT:
            connect_host = self.url.raw_host
            assert connect_host is not None
            if helpers.is_ipv6_address(connect_host):
                connect_host = f'[{connect_host}]'
            path = f'{connect_host}:{self.url.port}'
        elif self.proxy and (not self.is_ssl()):
            path = str(self.url)
        else:
            path = self.url.raw_path
            if self.url.raw_query_string:
                path += '?' + self.url.raw_query_string
        protocol = conn.protocol
        assert protocol is not None
        writer = StreamWriter(protocol, self.loop, on_chunk_sent=functools.partial(self._on_chunk_request_sent, self.method, self.url), on_headers_sent=functools.partial(self._on_headers_request_sent, self.method, self.url))
        if self.compress:
            writer.enable_compression(self.compress)
        if self.chunked is not None:
            writer.enable_chunking()
        if self.method in self.POST_METHODS and hdrs.CONTENT_TYPE not in self.skip_auto_headers and (hdrs.CONTENT_TYPE not in self.headers):
            self.headers[hdrs.CONTENT_TYPE] = 'application/octet-stream'
        connection = self.headers.get(hdrs.CONNECTION)
        if not connection:
            if self.keep_alive():
                if self.version == HttpVersion10:
                    connection = 'keep-alive'
            elif self.version == HttpVersion11:
                connection = 'close'
        if connection is not None:
            self.headers[hdrs.CONNECTION] = connection
        status_line = '{0} {1} HTTP/{v.major}.{v.minor}'.format(self.method, path, v=self.version)
        await writer.write_headers(status_line, self.headers)
        self._writer = self.loop.create_task(self.write_bytes(writer, conn))
        response_class = self.response_class
        assert response_class is not None
        self.response = response_class(self.method, self.original_url, writer=self._writer, continue100=self._continue, timer=self._timer, request_info=self.request_info, traces=self._traces, loop=self.loop, session=self._session)
        return self.response

    async def close(self) -> None:
        if self._writer is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._writer

    def terminate(self) -> None:
        if self._writer is not None:
            if not self.loop.is_closed():
                self._writer.cancel()
            self._writer.remove_done_callback(self.__reset_writer)
            self._writer = None

    async def _on_chunk_request_sent(self, method: str, url: URL, chunk: bytes) -> None:
        for trace in self._traces:
            await trace.send_request_chunk_sent(method, url, chunk)

    async def _on_headers_request_sent(self, method: str, url: URL, headers: 'CIMultiDict[str]') -> None:
        for trace in self._traces:
            await trace.send_request_headers(method, url, headers)