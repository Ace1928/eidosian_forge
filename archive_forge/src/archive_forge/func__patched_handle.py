from __future__ import absolute_import
from sentry_sdk.hub import Hub
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk._types import TYPE_CHECKING
def _patched_handle(self, environ):
    hub = Hub.current
    integration = hub.get_integration(BottleIntegration)
    if integration is None:
        return old_handle(self, environ)
    scope_manager = hub.push_scope()
    with scope_manager:
        app = self
        with hub.configure_scope() as scope:
            scope._name = 'bottle'
            scope.add_event_processor(_make_request_event_processor(app, bottle_request, integration))
        res = old_handle(self, environ)
    return res