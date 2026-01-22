import weakref
import contextlib
from inspect import iscoroutinefunction
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import (
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk._compat import iteritems
from sentry_sdk._types import TYPE_CHECKING
class TornadoRequestExtractor(RequestExtractor):

    def content_length(self):
        if self.request.body is None:
            return 0
        return len(self.request.body)

    def cookies(self):
        return {k: v.value for k, v in iteritems(self.request.cookies)}

    def raw_data(self):
        return self.request.body

    def form(self):
        return {k: [v.decode('latin1', 'replace') for v in vs] for k, vs in iteritems(self.request.body_arguments)}

    def is_json(self):
        return _is_json_content_type(self.request.headers.get('content-type'))

    def files(self):
        return {k: v[0] for k, v in iteritems(self.request.files) if v}

    def size_of_file(self, file):
        return len(file.body or ())