from __future__ import absolute_import
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
class FlaskIntegration(Integration):
    identifier = 'flask'
    transaction_style = ''

    def __init__(self, transaction_style='endpoint'):
        if transaction_style not in TRANSACTION_STYLE_VALUES:
            raise ValueError('Invalid value for transaction_style: %s (must be in %s)' % (transaction_style, TRANSACTION_STYLE_VALUES))
        self.transaction_style = transaction_style

    @staticmethod
    def setup_once():
        version = package_version('flask')
        if version is None:
            raise DidNotEnable('Unparsable Flask version.')
        if version < (0, 10):
            raise DidNotEnable('Flask 0.10 or newer is required.')
        before_render_template.connect(_add_sentry_trace)
        request_started.connect(_request_started)
        got_request_exception.connect(_capture_exception)
        old_app = Flask.__call__

        def sentry_patched_wsgi_app(self, environ, start_response):
            if Hub.current.get_integration(FlaskIntegration) is None:
                return old_app(self, environ, start_response)
            return SentryWsgiMiddleware(lambda *a, **kw: old_app(self, *a, **kw))(environ, start_response)
        Flask.__call__ = sentry_patched_wsgi_app