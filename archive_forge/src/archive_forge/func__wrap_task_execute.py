from __future__ import absolute_import
import sys
from datetime import datetime
from sentry_sdk._compat import reraise
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk import Hub
from sentry_sdk.api import continue_trace, get_baggage, get_traceparent
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
def _wrap_task_execute(func):

    def _sentry_execute(*args, **kwargs):
        hub = Hub.current
        if hub.get_integration(HueyIntegration) is None:
            return func(*args, **kwargs)
        try:
            result = func(*args, **kwargs)
        except Exception:
            exc_info = sys.exc_info()
            _capture_exception(exc_info)
            reraise(*exc_info)
        return result
    return _sentry_execute