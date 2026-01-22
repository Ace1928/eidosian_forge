from __future__ import absolute_import
from sentry_sdk.hub import Hub
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk._types import TYPE_CHECKING
class BottleRequestExtractor(RequestExtractor):

    def env(self):
        return self.request.environ

    def cookies(self):
        return self.request.cookies

    def raw_data(self):
        return self.request.body.read()

    def form(self):
        if self.is_json():
            return None
        return self.request.forms.decode()

    def files(self):
        if self.is_json():
            return None
        return self.request.files

    def size_of_file(self, file):
        return file.content_length