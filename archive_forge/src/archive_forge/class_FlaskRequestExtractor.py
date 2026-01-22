from __future__ import absolute_import
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
class FlaskRequestExtractor(RequestExtractor):

    def env(self):
        return self.request.environ

    def cookies(self):
        return {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in self.request.cookies.items()}

    def raw_data(self):
        return self.request.get_data()

    def form(self):
        return self.request.form

    def files(self):
        return self.request.files

    def is_json(self):
        return self.request.is_json

    def json(self):
        return self.request.get_json(silent=True)

    def size_of_file(self, file):
        return file.content_length