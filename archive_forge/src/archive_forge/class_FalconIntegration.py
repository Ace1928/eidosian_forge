from __future__ import absolute_import
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
class FalconIntegration(Integration):
    identifier = 'falcon'
    transaction_style = ''

    def __init__(self, transaction_style='uri_template'):
        if transaction_style not in TRANSACTION_STYLE_VALUES:
            raise ValueError('Invalid value for transaction_style: %s (must be in %s)' % (transaction_style, TRANSACTION_STYLE_VALUES))
        self.transaction_style = transaction_style

    @staticmethod
    def setup_once():
        version = parse_version(FALCON_VERSION)
        if version is None:
            raise DidNotEnable('Unparsable Falcon version: {}'.format(FALCON_VERSION))
        if version < (1, 4):
            raise DidNotEnable('Falcon 1.4 or newer required.')
        _patch_wsgi_app()
        _patch_handle_exception()
        _patch_prepare_middleware()