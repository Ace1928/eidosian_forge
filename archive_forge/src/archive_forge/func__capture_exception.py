from __future__ import absolute_import
import sys
from sentry_sdk import configure_scope
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def _capture_exception(exc_info, hub):
    client = hub.client
    client_options = client.options
    mechanism = {'type': 'spark', 'handled': False}
    exc_info = exc_info_from_error(exc_info)
    exc_type, exc_value, tb = exc_info
    rv = []
    for exc_type, exc_value, tb in walk_exception_chain(exc_info):
        if exc_type not in (SystemExit, EOFError, ConnectionResetError):
            rv.append(single_exception_from_error_tuple(exc_type, exc_value, tb, client_options, mechanism))
    if rv:
        rv.reverse()
        hint = event_hint_with_exc_info(exc_info)
        event = {'level': 'error', 'exception': {'values': rv}}
        _tag_task_context()
        hub.capture_event(event, hint=hint)