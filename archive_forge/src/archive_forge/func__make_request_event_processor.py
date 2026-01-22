from __future__ import absolute_import
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
def _make_request_event_processor(app, request, integration):

    def inner(event, hint):
        if request is None:
            return event
        with capture_internal_exceptions():
            FlaskRequestExtractor(request).extract_into_event(event)
        if _should_send_default_pii():
            with capture_internal_exceptions():
                _add_user_to_event(event)
        return event
    return inner