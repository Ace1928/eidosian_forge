from __future__ import absolute_import
import asyncio
import functools
import aiohttp  # type: ignore
import six
import urllib3  # type: ignore
from google.auth import exceptions
from google.auth import transport
from google.auth.transport import requests
class AuthorizedSession(aiohttp.ClientSession):
    """This is an async implementation of the Authorized Session class. We utilize an
    aiohttp transport instance, and the interface mirrors the google.auth.transport.requests
    Authorized Session class, except for the change in the transport used in the async use case.

    A Requests Session class with credentials.

    This class is used to perform requests to API endpoints that require
    authorization::

        from google.auth.transport import aiohttp_requests

        async with aiohttp_requests.AuthorizedSession(credentials) as authed_session:
            response = await authed_session.request(
                'GET', 'https://www.googleapis.com/storage/v1/b')

    The underlying :meth:`request` implementation handles adding the
    credentials' headers to the request and refreshing credentials as needed.

    Args:
        credentials (google.auth._credentials_async.Credentials):
            The credentials to add to the request.
        refresh_status_codes (Sequence[int]): Which HTTP status codes indicate
            that credentials should be refreshed and the request should be
            retried.
        max_refresh_attempts (int): The maximum number of times to attempt to
            refresh the credentials and retry the request.
        refresh_timeout (Optional[int]): The timeout value in seconds for
            credential refresh HTTP requests.
        auth_request (google.auth.transport.aiohttp_requests.Request):
            (Optional) An instance of
            :class:`~google.auth.transport.aiohttp_requests.Request` used when
            refreshing credentials. If not passed,
            an instance of :class:`~google.auth.transport.aiohttp_requests.Request`
            is created.
        kwargs: Additional arguments passed through to the underlying
            ClientSession :meth:`aiohttp.ClientSession` object.
    """

    def __init__(self, credentials, refresh_status_codes=transport.DEFAULT_REFRESH_STATUS_CODES, max_refresh_attempts=transport.DEFAULT_MAX_REFRESH_ATTEMPTS, refresh_timeout=None, auth_request=None, auto_decompress=False, **kwargs):
        super(AuthorizedSession, self).__init__(**kwargs)
        self.credentials = credentials
        self._refresh_status_codes = refresh_status_codes
        self._max_refresh_attempts = max_refresh_attempts
        self._refresh_timeout = refresh_timeout
        self._is_mtls = False
        self._auth_request = auth_request
        self._auth_request_session = None
        self._loop = asyncio.get_event_loop()
        self._refresh_lock = asyncio.Lock()
        self._auto_decompress = auto_decompress

    async def request(self, method, url, data=None, headers=None, max_allowed_time=None, timeout=_DEFAULT_TIMEOUT, auto_decompress=False, **kwargs):
        """Implementation of Authorized Session aiohttp request.

        Args:
            method (str):
                The http request method used (e.g. GET, PUT, DELETE)
            url (str):
                The url at which the http request is sent.
            data (Optional[dict]): Dictionary, list of tuples, bytes, or file-like
                object to send in the body of the Request.
            headers (Optional[dict]): Dictionary of HTTP Headers to send with the
                Request.
            timeout (Optional[Union[float, aiohttp.ClientTimeout]]):
                The amount of time in seconds to wait for the server response
                with each individual request. Can also be passed as an
                ``aiohttp.ClientTimeout`` object.
            max_allowed_time (Optional[float]):
                If the method runs longer than this, a ``Timeout`` exception is
                automatically raised. Unlike the ``timeout`` parameter, this
                value applies to the total method execution time, even if
                multiple requests are made under the hood.

                Mind that it is not guaranteed that the timeout error is raised
                at ``max_allowed_time``. It might take longer, for example, if
                an underlying request takes a lot of time, but the request
                itself does not timeout, e.g. if a large file is being
                transmitted. The timout error will be raised after such
                request completes.
        """
        if headers:
            for key in headers.keys():
                if type(headers[key]) is bytes:
                    headers[key] = headers[key].decode('utf-8')
        async with aiohttp.ClientSession(auto_decompress=self._auto_decompress) as self._auth_request_session:
            auth_request = Request(self._auth_request_session)
            self._auth_request = auth_request
            _credential_refresh_attempt = kwargs.pop('_credential_refresh_attempt', 0)
            request_headers = headers.copy() if headers is not None else {}
            auth_request = self._auth_request if timeout is None else functools.partial(self._auth_request, timeout=timeout)
            remaining_time = max_allowed_time
            with requests.TimeoutGuard(remaining_time, asyncio.TimeoutError) as guard:
                await self.credentials.before_request(auth_request, method, url, request_headers)
            with requests.TimeoutGuard(remaining_time, asyncio.TimeoutError) as guard:
                response = await super(AuthorizedSession, self).request(method, url, data=data, headers=request_headers, timeout=timeout, **kwargs)
            remaining_time = guard.remaining_timeout
            if response.status in self._refresh_status_codes and _credential_refresh_attempt < self._max_refresh_attempts:
                requests._LOGGER.info('Refreshing credentials due to a %s response. Attempt %s/%s.', response.status, _credential_refresh_attempt + 1, self._max_refresh_attempts)
                auth_request = self._auth_request if timeout is None else functools.partial(self._auth_request, timeout=timeout)
                with requests.TimeoutGuard(remaining_time, asyncio.TimeoutError) as guard:
                    async with self._refresh_lock:
                        await self._loop.run_in_executor(None, self.credentials.refresh, auth_request)
                remaining_time = guard.remaining_timeout
                return await self.request(method, url, data=data, headers=headers, max_allowed_time=remaining_time, timeout=timeout, _credential_refresh_attempt=_credential_refresh_attempt + 1, **kwargs)
        return response