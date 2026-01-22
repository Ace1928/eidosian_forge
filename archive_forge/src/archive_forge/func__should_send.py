import sys
from sentry_sdk.hub import Hub
from sentry_sdk.utils import capture_internal_exceptions, event_from_exception
from sentry_sdk.integrations import Integration
from sentry_sdk._types import TYPE_CHECKING
def _should_send(always_run=False):
    if always_run:
        return True
    if hasattr(sys, 'ps1'):
        return False
    return True