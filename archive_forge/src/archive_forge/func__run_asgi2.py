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
def _run_asgi2(self, scope):

    async def inner(receive, send):
        return await self._run_app(scope, receive, send, asgi_version=2)
    return inner