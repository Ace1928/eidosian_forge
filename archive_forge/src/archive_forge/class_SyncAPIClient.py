from __future__ import annotations
import json
import time
import uuid
import email
import asyncio
import inspect
import logging
import platform
import warnings
import email.utils
from types import TracebackType
from random import random
from typing import (
from functools import lru_cache
from typing_extensions import Literal, override, get_origin
import anyio
import httpx
import distro
import pydantic
from httpx import URL, Limits
from pydantic import PrivateAttr
from . import _exceptions
from ._qs import Querystring
from ._files import to_httpx_files, async_to_httpx_files
from ._types import (
from ._utils import is_dict, is_list, is_given, is_mapping
from ._compat import model_copy, model_dump
from ._models import GenericModel, FinalRequestOptions, validate_type, construct_type
from ._response import (
from ._constants import (
from ._streaming import Stream, SSEDecoder, AsyncStream, SSEBytesDecoder
from ._exceptions import (
from ._legacy_response import LegacyAPIResponse
class SyncAPIClient(BaseClient[httpx.Client, Stream[Any]]):
    _client: httpx.Client
    _default_stream_cls: type[Stream[Any]] | None = None

    def __init__(self, *, version: str, base_url: str | URL, max_retries: int=DEFAULT_MAX_RETRIES, timeout: float | Timeout | None | NotGiven=NOT_GIVEN, transport: Transport | None=None, proxies: ProxiesTypes | None=None, limits: Limits | None=None, http_client: httpx.Client | None=None, custom_headers: Mapping[str, str] | None=None, custom_query: Mapping[str, object] | None=None, _strict_response_validation: bool) -> None:
        if limits is not None:
            warnings.warn('The `connection_pool_limits` argument is deprecated. The `http_client` argument should be passed instead', category=DeprecationWarning, stacklevel=3)
            if http_client is not None:
                raise ValueError('The `http_client` argument is mutually exclusive with `connection_pool_limits`')
        else:
            limits = DEFAULT_LIMITS
        if transport is not None:
            warnings.warn('The `transport` argument is deprecated. The `http_client` argument should be passed instead', category=DeprecationWarning, stacklevel=3)
            if http_client is not None:
                raise ValueError('The `http_client` argument is mutually exclusive with `transport`')
        if proxies is not None:
            warnings.warn('The `proxies` argument is deprecated. The `http_client` argument should be passed instead', category=DeprecationWarning, stacklevel=3)
            if http_client is not None:
                raise ValueError('The `http_client` argument is mutually exclusive with `proxies`')
        if not is_given(timeout):
            if http_client and http_client.timeout != HTTPX_DEFAULT_TIMEOUT:
                timeout = http_client.timeout
            else:
                timeout = DEFAULT_TIMEOUT
        if http_client is not None and (not isinstance(http_client, httpx.Client)):
            raise TypeError(f'Invalid `http_client` argument; Expected an instance of `httpx.Client` but got {type(http_client)}')
        super().__init__(version=version, limits=limits, timeout=cast(Timeout, timeout), proxies=proxies, base_url=base_url, transport=transport, max_retries=max_retries, custom_query=custom_query, custom_headers=custom_headers, _strict_response_validation=_strict_response_validation)
        self._client = http_client or SyncHttpxClientWrapper(base_url=base_url, timeout=cast(Timeout, timeout), proxies=proxies, transport=transport, limits=limits, follow_redirects=True)

    def is_closed(self) -> bool:
        return self._client.is_closed

    def close(self) -> None:
        """Close the underlying HTTPX client.

        The client will *not* be usable after this.
        """
        if hasattr(self, '_client'):
            self._client.close()

    def __enter__(self: _T) -> _T:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, exc_tb: TracebackType | None) -> None:
        self.close()

    def _prepare_options(self, options: FinalRequestOptions) -> None:
        """Hook for mutating the given options"""
        return None

    def _prepare_request(self, request: httpx.Request) -> None:
        """This method is used as a callback for mutating the `Request` object
        after it has been constructed.
        This is useful for cases where you want to add certain headers based off of
        the request properties, e.g. `url`, `method` etc.
        """
        return None

    @overload
    def request(self, cast_to: Type[ResponseT], options: FinalRequestOptions, remaining_retries: Optional[int]=None, *, stream: Literal[True], stream_cls: Type[_StreamT]) -> _StreamT:
        ...

    @overload
    def request(self, cast_to: Type[ResponseT], options: FinalRequestOptions, remaining_retries: Optional[int]=None, *, stream: Literal[False]=False) -> ResponseT:
        ...

    @overload
    def request(self, cast_to: Type[ResponseT], options: FinalRequestOptions, remaining_retries: Optional[int]=None, *, stream: bool=False, stream_cls: Type[_StreamT] | None=None) -> ResponseT | _StreamT:
        ...

    def request(self, cast_to: Type[ResponseT], options: FinalRequestOptions, remaining_retries: Optional[int]=None, *, stream: bool=False, stream_cls: type[_StreamT] | None=None) -> ResponseT | _StreamT:
        return self._request(cast_to=cast_to, options=options, stream=stream, stream_cls=stream_cls, remaining_retries=remaining_retries)

    def _request(self, *, cast_to: Type[ResponseT], options: FinalRequestOptions, remaining_retries: int | None, stream: bool, stream_cls: type[_StreamT] | None) -> ResponseT | _StreamT:
        cast_to = self._maybe_override_cast_to(cast_to, options)
        self._prepare_options(options)
        retries = self._remaining_retries(remaining_retries, options)
        request = self._build_request(options)
        self._prepare_request(request)
        kwargs: HttpxSendArgs = {}
        if self.custom_auth is not None:
            kwargs['auth'] = self.custom_auth
        try:
            response = self._client.send(request, stream=stream or self._should_stream_response_body(request=request), **kwargs)
        except httpx.TimeoutException as err:
            log.debug('Encountered httpx.TimeoutException', exc_info=True)
            if retries > 0:
                return self._retry_request(options, cast_to, retries, stream=stream, stream_cls=stream_cls, response_headers=None)
            log.debug('Raising timeout error')
            raise APITimeoutError(request=request) from err
        except Exception as err:
            log.debug('Encountered Exception', exc_info=True)
            if retries > 0:
                return self._retry_request(options, cast_to, retries, stream=stream, stream_cls=stream_cls, response_headers=None)
            log.debug('Raising connection error')
            raise APIConnectionError(request=request) from err
        log.debug('HTTP Request: %s %s "%i %s"', request.method, request.url, response.status_code, response.reason_phrase)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            log.debug('Encountered httpx.HTTPStatusError', exc_info=True)
            if retries > 0 and self._should_retry(err.response):
                err.response.close()
                return self._retry_request(options, cast_to, retries, err.response.headers, stream=stream, stream_cls=stream_cls)
            if not err.response.is_closed:
                err.response.read()
            log.debug('Re-raising status error')
            raise self._make_status_error_from_response(err.response) from None
        return self._process_response(cast_to=cast_to, options=options, response=response, stream=stream, stream_cls=stream_cls)

    def _retry_request(self, options: FinalRequestOptions, cast_to: Type[ResponseT], remaining_retries: int, response_headers: httpx.Headers | None, *, stream: bool, stream_cls: type[_StreamT] | None) -> ResponseT | _StreamT:
        remaining = remaining_retries - 1
        if remaining == 1:
            log.debug('1 retry left')
        else:
            log.debug('%i retries left', remaining)
        timeout = self._calculate_retry_timeout(remaining, options, response_headers)
        log.info('Retrying request to %s in %f seconds', options.url, timeout)
        time.sleep(timeout)
        return self._request(options=options, cast_to=cast_to, remaining_retries=remaining, stream=stream, stream_cls=stream_cls)

    def _process_response(self, *, cast_to: Type[ResponseT], options: FinalRequestOptions, response: httpx.Response, stream: bool, stream_cls: type[Stream[Any]] | type[AsyncStream[Any]] | None) -> ResponseT:
        if response.request.headers.get(RAW_RESPONSE_HEADER) == 'true':
            return cast(ResponseT, LegacyAPIResponse(raw=response, client=self, cast_to=cast_to, stream=stream, stream_cls=stream_cls, options=options))
        origin = get_origin(cast_to) or cast_to
        if inspect.isclass(origin) and issubclass(origin, BaseAPIResponse):
            if not issubclass(origin, APIResponse):
                raise TypeError(f'API Response types must subclass {APIResponse}; Received {origin}')
            response_cls = cast('type[BaseAPIResponse[Any]]', cast_to)
            return cast(ResponseT, response_cls(raw=response, client=self, cast_to=extract_response_type(response_cls), stream=stream, stream_cls=stream_cls, options=options))
        if cast_to == httpx.Response:
            return cast(ResponseT, response)
        api_response = APIResponse(raw=response, client=self, cast_to=cast('type[ResponseT]', cast_to), stream=stream, stream_cls=stream_cls, options=options)
        if bool(response.request.headers.get(RAW_RESPONSE_HEADER)):
            return cast(ResponseT, api_response)
        return api_response.parse()

    def _request_api_list(self, model: Type[object], page: Type[SyncPageT], options: FinalRequestOptions) -> SyncPageT:

        def _parser(resp: SyncPageT) -> SyncPageT:
            resp._set_private_attributes(client=self, model=model, options=options)
            return resp
        options.post_parser = _parser
        return self.request(page, options, stream=False)

    @overload
    def get(self, path: str, *, cast_to: Type[ResponseT], options: RequestOptions={}, stream: Literal[False]=False) -> ResponseT:
        ...

    @overload
    def get(self, path: str, *, cast_to: Type[ResponseT], options: RequestOptions={}, stream: Literal[True], stream_cls: type[_StreamT]) -> _StreamT:
        ...

    @overload
    def get(self, path: str, *, cast_to: Type[ResponseT], options: RequestOptions={}, stream: bool, stream_cls: type[_StreamT] | None=None) -> ResponseT | _StreamT:
        ...

    def get(self, path: str, *, cast_to: Type[ResponseT], options: RequestOptions={}, stream: bool=False, stream_cls: type[_StreamT] | None=None) -> ResponseT | _StreamT:
        opts = FinalRequestOptions.construct(method='get', url=path, **options)
        return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))

    @overload
    def post(self, path: str, *, cast_to: Type[ResponseT], body: Body | None=None, options: RequestOptions={}, files: RequestFiles | None=None, stream: Literal[False]=False) -> ResponseT:
        ...

    @overload
    def post(self, path: str, *, cast_to: Type[ResponseT], body: Body | None=None, options: RequestOptions={}, files: RequestFiles | None=None, stream: Literal[True], stream_cls: type[_StreamT]) -> _StreamT:
        ...

    @overload
    def post(self, path: str, *, cast_to: Type[ResponseT], body: Body | None=None, options: RequestOptions={}, files: RequestFiles | None=None, stream: bool, stream_cls: type[_StreamT] | None=None) -> ResponseT | _StreamT:
        ...

    def post(self, path: str, *, cast_to: Type[ResponseT], body: Body | None=None, options: RequestOptions={}, files: RequestFiles | None=None, stream: bool=False, stream_cls: type[_StreamT] | None=None) -> ResponseT | _StreamT:
        opts = FinalRequestOptions.construct(method='post', url=path, json_data=body, files=to_httpx_files(files), **options)
        return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))

    def patch(self, path: str, *, cast_to: Type[ResponseT], body: Body | None=None, options: RequestOptions={}) -> ResponseT:
        opts = FinalRequestOptions.construct(method='patch', url=path, json_data=body, **options)
        return self.request(cast_to, opts)

    def put(self, path: str, *, cast_to: Type[ResponseT], body: Body | None=None, files: RequestFiles | None=None, options: RequestOptions={}) -> ResponseT:
        opts = FinalRequestOptions.construct(method='put', url=path, json_data=body, files=to_httpx_files(files), **options)
        return self.request(cast_to, opts)

    def delete(self, path: str, *, cast_to: Type[ResponseT], body: Body | None=None, options: RequestOptions={}) -> ResponseT:
        opts = FinalRequestOptions.construct(method='delete', url=path, json_data=body, **options)
        return self.request(cast_to, opts)

    def get_api_list(self, path: str, *, model: Type[object], page: Type[SyncPageT], body: Body | None=None, options: RequestOptions={}, method: str='get') -> SyncPageT:
        opts = FinalRequestOptions.construct(method=method, url=path, json_data=body, **options)
        return self._request_api_list(model, page, opts)