import datetime
import functools
from io import BytesIO
import ssl
import time
import weakref
from tornado.concurrent import (
from tornado.escape import utf8, native_str
from tornado import gen, httputil
from tornado.ioloop import IOLoop
from tornado.util import Configurable
from typing import Type, Any, Union, Dict, Callable, Optional, cast
class AsyncHTTPClient(Configurable):
    """An non-blocking HTTP client.

    Example usage::

        async def f():
            http_client = AsyncHTTPClient()
            try:
                response = await http_client.fetch("http://www.google.com")
            except Exception as e:
                print("Error: %s" % e)
            else:
                print(response.body)

    The constructor for this class is magic in several respects: It
    actually creates an instance of an implementation-specific
    subclass, and instances are reused as a kind of pseudo-singleton
    (one per `.IOLoop`). The keyword argument ``force_instance=True``
    can be used to suppress this singleton behavior. Unless
    ``force_instance=True`` is used, no arguments should be passed to
    the `AsyncHTTPClient` constructor. The implementation subclass as
    well as arguments to its constructor can be set with the static
    method `configure()`

    All `AsyncHTTPClient` implementations support a ``defaults``
    keyword argument, which can be used to set default values for
    `HTTPRequest` attributes.  For example::

        AsyncHTTPClient.configure(
            None, defaults=dict(user_agent="MyUserAgent"))
        # or with force_instance:
        client = AsyncHTTPClient(force_instance=True,
            defaults=dict(user_agent="MyUserAgent"))

    .. versionchanged:: 5.0
       The ``io_loop`` argument (deprecated since version 4.1) has been removed.

    """
    _instance_cache = None

    @classmethod
    def configurable_base(cls) -> Type[Configurable]:
        return AsyncHTTPClient

    @classmethod
    def configurable_default(cls) -> Type[Configurable]:
        from tornado.simple_httpclient import SimpleAsyncHTTPClient
        return SimpleAsyncHTTPClient

    @classmethod
    def _async_clients(cls) -> Dict[IOLoop, 'AsyncHTTPClient']:
        attr_name = '_async_client_dict_' + cls.__name__
        if not hasattr(cls, attr_name):
            setattr(cls, attr_name, weakref.WeakKeyDictionary())
        return getattr(cls, attr_name)

    def __new__(cls, force_instance: bool=False, **kwargs: Any) -> 'AsyncHTTPClient':
        io_loop = IOLoop.current()
        if force_instance:
            instance_cache = None
        else:
            instance_cache = cls._async_clients()
        if instance_cache is not None and io_loop in instance_cache:
            return instance_cache[io_loop]
        instance = super(AsyncHTTPClient, cls).__new__(cls, **kwargs)
        instance._instance_cache = instance_cache
        if instance_cache is not None:
            instance_cache[instance.io_loop] = instance
        return instance

    def initialize(self, defaults: Optional[Dict[str, Any]]=None) -> None:
        self.io_loop = IOLoop.current()
        self.defaults = dict(HTTPRequest._DEFAULTS)
        if defaults is not None:
            self.defaults.update(defaults)
        self._closed = False

    def close(self) -> None:
        """Destroys this HTTP client, freeing any file descriptors used.

        This method is **not needed in normal use** due to the way
        that `AsyncHTTPClient` objects are transparently reused.
        ``close()`` is generally only necessary when either the
        `.IOLoop` is also being closed, or the ``force_instance=True``
        argument was used when creating the `AsyncHTTPClient`.

        No other methods may be called on the `AsyncHTTPClient` after
        ``close()``.

        """
        if self._closed:
            return
        self._closed = True
        if self._instance_cache is not None:
            cached_val = self._instance_cache.pop(self.io_loop, None)
            if cached_val is not None and cached_val is not self:
                raise RuntimeError('inconsistent AsyncHTTPClient cache')

    def fetch(self, request: Union[str, 'HTTPRequest'], raise_error: bool=True, **kwargs: Any) -> 'Future[HTTPResponse]':
        """Executes a request, asynchronously returning an `HTTPResponse`.

        The request may be either a string URL or an `HTTPRequest` object.
        If it is a string, we construct an `HTTPRequest` using any additional
        kwargs: ``HTTPRequest(request, **kwargs)``

        This method returns a `.Future` whose result is an
        `HTTPResponse`. By default, the ``Future`` will raise an
        `HTTPError` if the request returned a non-200 response code
        (other errors may also be raised if the server could not be
        contacted). Instead, if ``raise_error`` is set to False, the
        response will always be returned regardless of the response
        code.

        If a ``callback`` is given, it will be invoked with the `HTTPResponse`.
        In the callback interface, `HTTPError` is not automatically raised.
        Instead, you must check the response's ``error`` attribute or
        call its `~HTTPResponse.rethrow` method.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.

           The ``raise_error=False`` argument only affects the
           `HTTPError` raised when a non-200 response code is used,
           instead of suppressing all errors.
        """
        if self._closed:
            raise RuntimeError('fetch() called on closed AsyncHTTPClient')
        if not isinstance(request, HTTPRequest):
            request = HTTPRequest(url=request, **kwargs)
        elif kwargs:
            raise ValueError("kwargs can't be used if request is an HTTPRequest object")
        request.headers = httputil.HTTPHeaders(request.headers)
        request_proxy = _RequestProxy(request, self.defaults)
        future = Future()

        def handle_response(response: 'HTTPResponse') -> None:
            if response.error:
                if raise_error or not response._error_is_response_code:
                    future_set_exception_unless_cancelled(future, response.error)
                    return
            future_set_result_unless_cancelled(future, response)
        self.fetch_impl(cast(HTTPRequest, request_proxy), handle_response)
        return future

    def fetch_impl(self, request: 'HTTPRequest', callback: Callable[['HTTPResponse'], None]) -> None:
        raise NotImplementedError()

    @classmethod
    def configure(cls, impl: 'Union[None, str, Type[Configurable]]', **kwargs: Any) -> None:
        """Configures the `AsyncHTTPClient` subclass to use.

        ``AsyncHTTPClient()`` actually creates an instance of a subclass.
        This method may be called with either a class object or the
        fully-qualified name of such a class (or ``None`` to use the default,
        ``SimpleAsyncHTTPClient``)

        If additional keyword arguments are given, they will be passed
        to the constructor of each subclass instance created.  The
        keyword argument ``max_clients`` determines the maximum number
        of simultaneous `~AsyncHTTPClient.fetch()` operations that can
        execute in parallel on each `.IOLoop`.  Additional arguments
        may be supported depending on the implementation class in use.

        Example::

           AsyncHTTPClient.configure("tornado.curl_httpclient.CurlAsyncHTTPClient")
        """
        super(AsyncHTTPClient, cls).configure(impl, **kwargs)