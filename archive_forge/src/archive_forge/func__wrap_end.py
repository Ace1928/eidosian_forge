from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.tracing import Span
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import capture_internal_exceptions
from typing import TypeVar
def _wrap_end(f: Callable[P, T]) -> Callable[P, T]:

    def _inner_end(*args: P.args, **kwargs: P.kwargs) -> T:
        res = f(*args, **kwargs)
        instance = args[0]
        span = instance.connection._sentry_span
        if span is not None:
            if res is not None and _should_send_default_pii():
                span.set_data('db.result', res)
            with capture_internal_exceptions():
                span.hub.add_breadcrumb(message=span._data.pop('query'), category='query', data=span._data)
            span.finish()
        return res
    return _inner_end