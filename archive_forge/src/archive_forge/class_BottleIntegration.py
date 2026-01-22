from __future__ import absolute_import
from sentry_sdk.hub import Hub
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk._types import TYPE_CHECKING
class BottleIntegration(Integration):
    identifier = 'bottle'
    transaction_style = ''

    def __init__(self, transaction_style='endpoint'):
        if transaction_style not in TRANSACTION_STYLE_VALUES:
            raise ValueError('Invalid value for transaction_style: %s (must be in %s)' % (transaction_style, TRANSACTION_STYLE_VALUES))
        self.transaction_style = transaction_style

    @staticmethod
    def setup_once():
        version = parse_version(BOTTLE_VERSION)
        if version is None:
            raise DidNotEnable('Unparsable Bottle version: {}'.format(BOTTLE_VERSION))
        if version < (0, 12):
            raise DidNotEnable('Bottle 0.12 or newer required.')
        old_app = Bottle.__call__

        def sentry_patched_wsgi_app(self, environ, start_response):
            hub = Hub.current
            integration = hub.get_integration(BottleIntegration)
            if integration is None:
                return old_app(self, environ, start_response)
            return SentryWsgiMiddleware(lambda *a, **kw: old_app(self, *a, **kw))(environ, start_response)
        Bottle.__call__ = sentry_patched_wsgi_app
        old_handle = Bottle._handle

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
        Bottle._handle = _patched_handle
        old_make_callback = Route._make_callback

        def patched_make_callback(self, *args, **kwargs):
            hub = Hub.current
            integration = hub.get_integration(BottleIntegration)
            prepared_callback = old_make_callback(self, *args, **kwargs)
            if integration is None:
                return prepared_callback
            client = hub.client

            def wrapped_callback(*args, **kwargs):
                try:
                    res = prepared_callback(*args, **kwargs)
                except HTTPResponse:
                    raise
                except Exception as exception:
                    event, hint = event_from_exception(exception, client_options=client.options, mechanism={'type': 'bottle', 'handled': False})
                    hub.capture_event(event, hint=hint)
                    raise exception
                return res
            return wrapped_callback
        Route._make_callback = patched_make_callback