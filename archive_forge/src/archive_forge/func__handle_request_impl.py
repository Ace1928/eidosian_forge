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
@contextlib.contextmanager
def _handle_request_impl(self):
    hub = Hub.current
    integration = hub.get_integration(TornadoIntegration)
    if integration is None:
        yield
    weak_handler = weakref.ref(self)
    with Hub(hub) as hub:
        headers = self.request.headers
        with hub.configure_scope() as scope:
            scope.clear_breadcrumbs()
            processor = _make_event_processor(weak_handler)
            scope.add_event_processor(processor)
        transaction = continue_trace(headers, op=OP.HTTP_SERVER, name='generic Tornado request', source=TRANSACTION_SOURCE_ROUTE)
        with hub.start_transaction(transaction, custom_sampling_context={'tornado_request': self.request}):
            yield