import asyncio
import inspect
from copy import deepcopy
from sentry_sdk._functools import partial
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub
from sentry_sdk.integrations._asgi_common import (
from sentry_sdk.sessions import auto_session_tracking
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
from sentry_sdk.tracing import Transaction
def _looks_like_asgi3(app):
    """
    Try to figure out if an application object supports ASGI3.

    This is how uvicorn figures out the application version as well.
    """
    if inspect.isclass(app):
        return hasattr(app, '__await__')
    elif inspect.isfunction(app):
        return asyncio.iscoroutinefunction(app)
    else:
        call = getattr(app, '__call__', None)
        return asyncio.iscoroutinefunction(call)