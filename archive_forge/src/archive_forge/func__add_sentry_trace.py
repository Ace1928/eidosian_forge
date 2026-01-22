from __future__ import absolute_import
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
def _add_sentry_trace(sender, template, context, **extra):
    if 'sentry_trace' in context:
        return
    hub = Hub.current
    trace_meta = Markup(hub.trace_propagation_meta())
    context['sentry_trace'] = trace_meta
    context['sentry_trace_meta'] = trace_meta