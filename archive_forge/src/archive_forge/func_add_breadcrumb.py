import inspect
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import NoOpSpan, Transaction
@hubmethod
def add_breadcrumb(crumb=None, hint=None, **kwargs):
    return Hub.current.add_breadcrumb(crumb, hint, **kwargs)