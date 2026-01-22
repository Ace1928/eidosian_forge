import asyncio
from copy import deepcopy
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable
from sentry_sdk.tracing import SOURCE_FOR_STYLE, TRANSACTION_SOURCE_ROUTE
from sentry_sdk.utils import transaction_from_function, logger
@wraps(old_call)
def _sentry_call(*args, **kwargs):
    hub = Hub.current
    with hub.configure_scope() as sentry_scope:
        if sentry_scope.profile is not None:
            sentry_scope.profile.update_active_thread_id()
        return old_call(*args, **kwargs)