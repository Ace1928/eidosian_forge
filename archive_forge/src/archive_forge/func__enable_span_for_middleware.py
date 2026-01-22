from __future__ import absolute_import
import asyncio
import functools
from copy import deepcopy
from sentry_sdk._compat import iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import (
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
def _enable_span_for_middleware(middleware_class):
    old_call = middleware_class.__call__

    async def _create_span_call(app, scope, receive, send, **kwargs):
        hub = Hub.current
        integration = hub.get_integration(StarletteIntegration)
        if integration is not None:
            middleware_name = app.__class__.__name__
            with hub.configure_scope() as sentry_scope:
                name, source = _get_transaction_from_middleware(app, scope, integration)
                if name is not None:
                    sentry_scope.set_transaction_name(name, source=source)
            with hub.start_span(op=OP.MIDDLEWARE_STARLETTE, description=middleware_name) as middleware_span:
                middleware_span.set_tag('starlette.middleware_name', middleware_name)

                async def _sentry_receive(*args, **kwargs):
                    hub = Hub.current
                    with hub.start_span(op=OP.MIDDLEWARE_STARLETTE_RECEIVE, description=getattr(receive, '__qualname__', str(receive))) as span:
                        span.set_tag('starlette.middleware_name', middleware_name)
                        return await receive(*args, **kwargs)
                receive_name = getattr(receive, '__name__', str(receive))
                receive_patched = receive_name == '_sentry_receive'
                new_receive = _sentry_receive if not receive_patched else receive

                async def _sentry_send(*args, **kwargs):
                    hub = Hub.current
                    with hub.start_span(op=OP.MIDDLEWARE_STARLETTE_SEND, description=getattr(send, '__qualname__', str(send))) as span:
                        span.set_tag('starlette.middleware_name', middleware_name)
                        return await send(*args, **kwargs)
                send_name = getattr(send, '__name__', str(send))
                send_patched = send_name == '_sentry_send'
                new_send = _sentry_send if not send_patched else send
                return await old_call(app, scope, new_receive, new_send, **kwargs)
        else:
            return await old_call(app, scope, receive, send, **kwargs)
    not_yet_patched = old_call.__name__ not in ['_create_span_call', '_sentry_authenticationmiddleware_call', '_sentry_exceptionmiddleware_call']
    if not_yet_patched:
        middleware_class.__call__ = _create_span_call
    return middleware_class