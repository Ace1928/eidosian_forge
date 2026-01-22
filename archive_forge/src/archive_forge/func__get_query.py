import urllib
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations._wsgi_common import _filter_headers
from sentry_sdk._types import TYPE_CHECKING
def _get_query(asgi_scope):
    """
    Extract querystring from the ASGI scope, in the format that the Sentry protocol expects.
    """
    qs = asgi_scope.get('query_string')
    if not qs:
        return None
    return urllib.parse.unquote(qs.decode('latin-1'))