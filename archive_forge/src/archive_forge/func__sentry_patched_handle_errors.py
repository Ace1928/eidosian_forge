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
def _sentry_patched_handle_errors(self, errors, response_data):
    hub = Hub.current
    integration = hub.get_integration(StrawberryIntegration)
    if integration is None:
        return
    if not errors:
        return
    with hub.configure_scope() as scope:
        event_processor = _make_response_event_processor(response_data)
        scope.add_event_processor(event_processor)
    with capture_internal_exceptions():
        for error in errors:
            event, hint = event_from_exception(error, client_options=hub.client.options if hub.client else None, mechanism={'type': integration.identifier, 'handled': False})
            hub.capture_event(event, hint=hint)