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
class ClientResponse(HeadersMixin):
    version: Optional[HttpVersion] = None
    status: int = None
    reason: Optional[str] = None
    content: StreamReader = None
    _headers: CIMultiDictProxy[str] = None
    _raw_headers: RawHeaders = None
    _connection = None
    _source_traceback: Optional[traceback.StackSummary] = None
    _closed = True
    _released = False
    __writer = None

    def __init__(self, method: str, url: URL, *, writer: 'asyncio.Task[None]', continue100: Optional['asyncio.Future[bool]'], timer: BaseTimerContext, request_info: RequestInfo, traces: List['Trace'], loop: asyncio.AbstractEventLoop, session: 'ClientSession') -> None:
        assert isinstance(url, URL)
        self.method = method
        self.cookies = SimpleCookie()
        self._real_url = url
        self._url = url.with_fragment(None)
        self._body: Any = None
        self._writer: Optional[asyncio.Task[None]] = writer
        self._continue = continue100
        self._closed = True
        self._history: Tuple[ClientResponse, ...] = ()
        self._request_info = request_info
        self._timer = timer if timer is not None else TimerNoop()
        self._cache: Dict[str, Any] = {}
        self._traces = traces
        self._loop = loop
        self._session: Optional[ClientSession] = session
        if session is None:
            self._resolve_charset: Callable[['ClientResponse', bytes], str] = lambda *_: 'utf-8'
        else:
            self._resolve_charset = session._resolve_charset
        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(1))

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

    @reify
    def url(self) -> URL:
        return self._url

    @reify
    def url_obj(self) -> URL:
        warnings.warn('Deprecated, use .url #1654', DeprecationWarning, stacklevel=2)
        return self._url

    @reify
    def real_url(self) -> URL:
        return self._real_url

    @reify
    def host(self) -> str:
        assert self._url.host is not None
        return self._url.host

    @reify
    def headers(self) -> 'CIMultiDictProxy[str]':
        return self._headers

    @reify
    def raw_headers(self) -> RawHeaders:
        return self._raw_headers

    @reify
    def request_info(self) -> RequestInfo:
        return self._request_info

    @reify
    def content_disposition(self) -> Optional[ContentDisposition]:
        raw = self._headers.get(hdrs.CONTENT_DISPOSITION)
        if raw is None:
            return None
        disposition_type, params_dct = multipart.parse_content_disposition(raw)
        params = MappingProxyType(params_dct)
        filename = multipart.content_disposition_filename(params)
        return ContentDisposition(disposition_type, params, filename)

    def __del__(self, _warnings: Any=warnings) -> None:
        if self._closed:
            return
        if self._connection is not None:
            self._connection.release()
            self._cleanup_writer()
            if self._loop.get_debug():
                kwargs = {'source': self}
                _warnings.warn(f'Unclosed response {self!r}', ResourceWarning, **kwargs)
                context = {'client_response': self, 'message': 'Unclosed response'}
                if self._source_traceback:
                    context['source_traceback'] = self._source_traceback
                self._loop.call_exception_handler(context)

    def __repr__(self) -> str:
        out = io.StringIO()
        ascii_encodable_url = str(self.url)
        if self.reason:
            ascii_encodable_reason = self.reason.encode('ascii', 'backslashreplace').decode('ascii')
        else:
            ascii_encodable_reason = 'None'
        print('<ClientResponse({}) [{} {}]>'.format(ascii_encodable_url, self.status, ascii_encodable_reason), file=out)
        print(self.headers, file=out)
        return out.getvalue()

    @property
    def connection(self) -> Optional['Connection']:
        return self._connection

    @reify
    def history(self) -> Tuple['ClientResponse', ...]:
        """A sequence of of responses, if redirects occurred."""
        return self._history

    @reify
    def links(self) -> 'MultiDictProxy[MultiDictProxy[Union[str, URL]]]':
        links_str = ', '.join(self.headers.getall('link', []))
        if not links_str:
            return MultiDictProxy(MultiDict())
        links: MultiDict[MultiDictProxy[Union[str, URL]]] = MultiDict()
        for val in re.split(',(?=\\s*<)', links_str):
            match = re.match('\\s*<(.*)>(.*)', val)
            if match is None:
                continue
            url, params_str = match.groups()
            params = params_str.split(';')[1:]
            link: MultiDict[Union[str, URL]] = MultiDict()
            for param in params:
                match = re.match('^\\s*(\\S*)\\s*=\\s*([\'\\"]?)(.*?)(\\2)\\s*$', param, re.M)
                if match is None:
                    continue
                key, _, value, _ = match.groups()
                link.add(key, value)
            key = link.get('rel', url)
            link.add('url', self.url.join(URL(url)))
            links.add(str(key), MultiDictProxy(link))
        return MultiDictProxy(links)

    async def start(self, connection: 'Connection') -> 'ClientResponse':
        """Start response processing."""
        self._closed = False
        self._protocol = connection.protocol
        self._connection = connection
        with self._timer:
            while True:
                try:
                    protocol = self._protocol
                    message, payload = await protocol.read()
                except http.HttpProcessingError as exc:
                    raise ClientResponseError(self.request_info, self.history, status=exc.code, message=exc.message, headers=exc.headers) from exc
                if message.code < 100 or message.code > 199 or message.code == 101:
                    break
                if self._continue is not None:
                    set_result(self._continue, True)
                    self._continue = None
        payload.on_eof(self._response_eof)
        self.version = message.version
        self.status = message.code
        self.reason = message.reason
        self._headers = message.headers
        self._raw_headers = message.raw_headers
        self.content = payload
        for hdr in self.headers.getall(hdrs.SET_COOKIE, ()):
            try:
                self.cookies.load(hdr)
            except CookieError as exc:
                client_logger.warning('Can not load response cookies: %s', exc)
        return self

    def _response_eof(self) -> None:
        if self._closed:
            return
        protocol = self._connection and self._connection.protocol
        if protocol is not None and protocol.upgraded:
            return
        self._closed = True
        self._cleanup_writer()
        self._release_connection()

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if not self._released:
            self._notify_content()
        self._closed = True
        if self._loop is None or self._loop.is_closed():
            return
        self._cleanup_writer()
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def release(self) -> Any:
        if not self._released:
            self._notify_content()
        self._closed = True
        self._cleanup_writer()
        self._release_connection()
        return noop()

    @property
    def ok(self) -> bool:
        """Returns ``True`` if ``status`` is less than ``400``, ``False`` if not.

        This is **not** a check for ``200 OK`` but a check that the response
        status is under 400.
        """
        return 400 > self.status

    def raise_for_status(self) -> None:
        if not self.ok:
            assert self.reason is not None
            self.release()
            raise ClientResponseError(self.request_info, self.history, status=self.status, message=self.reason, headers=self.headers)

    def _release_connection(self) -> None:
        if self._connection is not None:
            if self._writer is None:
                self._connection.release()
                self._connection = None
            else:
                self._writer.add_done_callback(lambda f: self._release_connection())

    async def _wait_released(self) -> None:
        if self._writer is not None:
            await self._writer
        self._release_connection()

    def _cleanup_writer(self) -> None:
        if self._writer is not None:
            self._writer.cancel()
        self._session = None

    def _notify_content(self) -> None:
        content = self.content
        if content and content.exception() is None:
            content.set_exception(ClientConnectionError('Connection closed'))
        self._released = True

    async def wait_for_close(self) -> None:
        if self._writer is not None:
            await self._writer
        self.release()

    async def read(self) -> bytes:
        """Read response payload."""
        if self._body is None:
            try:
                self._body = await self.content.read()
                for trace in self._traces:
                    await trace.send_response_chunk_received(self.method, self.url, self._body)
            except BaseException:
                self.close()
                raise
        elif self._released:
            raise ClientConnectionError('Connection closed')
        protocol = self._connection and self._connection.protocol
        if protocol is None or not protocol.upgraded:
            await self._wait_released()
        return self._body

    def get_encoding(self) -> str:
        ctype = self.headers.get(hdrs.CONTENT_TYPE, '').lower()
        mimetype = helpers.parse_mimetype(ctype)
        encoding = mimetype.parameters.get('charset')
        if encoding:
            with contextlib.suppress(LookupError):
                return codecs.lookup(encoding).name
        if mimetype.type == 'application' and (mimetype.subtype == 'json' or mimetype.subtype == 'rdap'):
            return 'utf-8'
        if self._body is None:
            raise RuntimeError('Cannot compute fallback encoding of a not yet read body')
        return self._resolve_charset(self, self._body)

    async def text(self, encoding: Optional[str]=None, errors: str='strict') -> str:
        """Read response payload and decode."""
        if self._body is None:
            await self.read()
        if encoding is None:
            encoding = self.get_encoding()
        return self._body.decode(encoding, errors=errors)

    async def json(self, *, encoding: Optional[str]=None, loads: JSONDecoder=DEFAULT_JSON_DECODER, content_type: Optional[str]='application/json') -> Any:
        """Read and decodes JSON response."""
        if self._body is None:
            await self.read()
        if content_type:
            ctype = self.headers.get(hdrs.CONTENT_TYPE, '').lower()
            if not _is_expected_content_type(ctype, content_type):
                raise ContentTypeError(self.request_info, self.history, message='Attempt to decode JSON with unexpected mimetype: %s' % ctype, headers=self.headers)
        stripped = self._body.strip()
        if not stripped:
            return None
        if encoding is None:
            encoding = self.get_encoding()
        return loads(stripped.decode(encoding))

    async def __aenter__(self) -> 'ClientResponse':
        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        self.release()
        await self.wait_for_close()