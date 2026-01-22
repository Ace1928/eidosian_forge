from __future__ import absolute_import
import asyncio
import functools
from copy import deepcopy
from sentry_sdk._compat import iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import (
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
def _get_transaction_from_middleware(app, asgi_scope, integration):
    name = None
    source = None
    if integration.transaction_style == 'endpoint':
        name = transaction_from_function(app.__class__)
        source = TRANSACTION_SOURCE_COMPONENT
    elif integration.transaction_style == 'url':
        name = _transaction_name_from_router(asgi_scope)
        source = TRANSACTION_SOURCE_ROUTE
    return (name, source)