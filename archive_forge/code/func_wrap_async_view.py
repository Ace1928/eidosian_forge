import asyncio
from django.core.handlers.wsgi import WSGIRequest
from sentry_sdk import Hub, _functools
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.utils import capture_internal_exceptions
def wrap_async_view(hub, callback):

    @_functools.wraps(callback)
    async def sentry_wrapped_callback(request, *args, **kwargs):
        with hub.configure_scope() as sentry_scope:
            if sentry_scope.profile is not None:
                sentry_scope.profile.update_active_thread_id()
            with hub.start_span(op=OP.VIEW_RENDER, description=request.resolver_match.view_name):
                return await callback(request, *args, **kwargs)
    return sentry_wrapped_callback