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
def _sentry_middleware_init(self, cls, **options):
    if cls == SentryAsgiMiddleware:
        return old_middleware_init(self, cls, **options)
    span_enabled_cls = _enable_span_for_middleware(cls)
    old_middleware_init(self, span_enabled_cls, **options)
    if cls == AuthenticationMiddleware:
        patch_authentication_middleware(cls)
    if cls == ExceptionMiddleware:
        patch_exception_middleware(cls)