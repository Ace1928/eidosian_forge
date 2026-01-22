from sentry_sdk.utils import event_from_exception, parse_version
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk._types import TYPE_CHECKING
def _transport_method(transport):
    """
    The RequestsHTTPTransport allows defining the HTTP method; all
    other transports use POST.
    """
    try:
        return transport.method
    except AttributeError:
        return 'POST'