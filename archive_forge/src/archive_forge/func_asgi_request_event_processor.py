import asyncio
from django.core.handlers.wsgi import WSGIRequest
from sentry_sdk import Hub, _functools
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.utils import capture_internal_exceptions
def asgi_request_event_processor(event, hint):
    from sentry_sdk.integrations.django import DjangoRequestExtractor, _set_user_info
    if request is None:
        return event
    if type(request) == WSGIRequest:
        return event
    with capture_internal_exceptions():
        DjangoRequestExtractor(request).extract_into_event(event)
    if _should_send_default_pii():
        with capture_internal_exceptions():
            _set_user_info(request, event)
    return event