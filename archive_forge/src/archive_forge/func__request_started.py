from __future__ import absolute_import
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
def _request_started(app, **kwargs):
    hub = Hub.current
    integration = hub.get_integration(FlaskIntegration)
    if integration is None:
        return
    with hub.configure_scope() as scope:
        request = flask_request._get_current_object()
        _set_transaction_name_and_source(scope, integration.transaction_style, request)
        evt_processor = _make_request_event_processor(app, request, integration)
        scope.add_event_processor(evt_processor)