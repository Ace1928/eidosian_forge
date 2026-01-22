from __future__ import absolute_import
import sys
from sentry_sdk._compat import reraise
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk import Hub
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.tracing import Transaction, TRANSACTION_SOURCE_TASK
from sentry_sdk.utils import (
def _wrap_coroutine(name, coroutine):

    async def _sentry_coroutine(ctx, *args, **kwargs):
        hub = Hub.current
        if hub.get_integration(ArqIntegration) is None:
            return await coroutine(ctx, *args, **kwargs)
        hub.scope.add_event_processor(_make_event_processor({**ctx, 'job_name': name}, *args, **kwargs))
        try:
            result = await coroutine(ctx, *args, **kwargs)
        except Exception:
            exc_info = sys.exc_info()
            _capture_exception(exc_info)
            reraise(*exc_info)
        return result
    return _sentry_coroutine