import sys
import weakref
from sentry_sdk.api import continue_trace
from sentry_sdk._compat import reraise
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.sessions import auto_session_tracking
from sentry_sdk.integrations._wsgi_common import (
from sentry_sdk.tracing import (
from sentry_sdk.tracing_utils import should_propagate_trace
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def _make_request_processor(weak_request):

    def aiohttp_processor(event, hint):
        request = weak_request()
        if request is None:
            return event
        with capture_internal_exceptions():
            request_info = event.setdefault('request', {})
            request_info['url'] = '%s://%s%s' % (request.scheme, request.host, request.path)
            request_info['query_string'] = request.query_string
            request_info['method'] = request.method
            request_info['env'] = {'REMOTE_ADDR': request.remote}
            hub = Hub.current
            request_info['headers'] = _filter_headers(dict(request.headers))
            request_info['data'] = get_aiohttp_request_data(hub, request)
        return event
    return aiohttp_processor