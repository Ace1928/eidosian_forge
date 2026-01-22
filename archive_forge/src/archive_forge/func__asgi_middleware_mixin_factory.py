import asyncio
from django.core.handlers.wsgi import WSGIRequest
from sentry_sdk import Hub, _functools
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.utils import capture_internal_exceptions
def _asgi_middleware_mixin_factory(_check_middleware_span):
    """
    Mixin class factory that generates a middleware mixin for handling requests
    in async mode.
    """

    class SentryASGIMixin:
        if TYPE_CHECKING:
            _inner = None

        def __init__(self, get_response):
            self.get_response = get_response
            self._acall_method = None
            self._async_check()

        def _async_check(self):
            """
            If get_response is a coroutine function, turns us into async mode so
            a thread is not consumed during a whole request.
            Taken from django.utils.deprecation::MiddlewareMixin._async_check
            """
            if asyncio.iscoroutinefunction(self.get_response):
                self._is_coroutine = asyncio.coroutines._is_coroutine

        def async_route_check(self):
            """
            Function that checks if we are in async mode,
            and if we are forwards the handling of requests to __acall__
            """
            return asyncio.iscoroutinefunction(self.get_response)

        async def __acall__(self, *args, **kwargs):
            f = self._acall_method
            if f is None:
                if hasattr(self._inner, '__acall__'):
                    self._acall_method = f = self._inner.__acall__
                else:
                    self._acall_method = f = self._inner
            middleware_span = _check_middleware_span(old_method=f)
            if middleware_span is None:
                return await f(*args, **kwargs)
            with middleware_span:
                return await f(*args, **kwargs)
    return SentryASGIMixin