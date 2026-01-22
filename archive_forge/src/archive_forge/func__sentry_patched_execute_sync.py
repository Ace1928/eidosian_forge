import hashlib
from functools import cached_property
from inspect import isawaitable
from sentry_sdk import configure_scope, start_span
from sentry_sdk.consts import OP
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def _sentry_patched_execute_sync(*args, **kwargs):
    hub = Hub.current
    integration = hub.get_integration(StrawberryIntegration)
    if integration is None:
        return old_execute_sync(*args, **kwargs)
    result = old_execute_sync(*args, **kwargs)
    if 'execution_context' in kwargs and result.errors:
        with hub.configure_scope() as scope:
            event_processor = _make_request_event_processor(kwargs['execution_context'])
            scope.add_event_processor(event_processor)
    return result