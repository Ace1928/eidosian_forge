import sys
from sentry_sdk._compat import reraise
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.aws_lambda import _make_request_event_processor
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk._functools import wraps
import chalice  # type: ignore
from chalice import Chalice, ChaliceViewError
from chalice.app import EventSourceHandler as ChaliceEventSourceHandler  # type: ignore
@wraps(view_function)
def wrapped_view_function(**function_args):
    hub = Hub.current
    client = hub.client
    with hub.push_scope() as scope:
        with capture_internal_exceptions():
            configured_time = app.lambda_context.get_remaining_time_in_millis()
            scope.set_transaction_name(app.lambda_context.function_name, source=TRANSACTION_SOURCE_COMPONENT)
            scope.add_event_processor(_make_request_event_processor(app.current_request.to_dict(), app.lambda_context, configured_time))
        try:
            return view_function(**function_args)
        except Exception as exc:
            if isinstance(exc, ChaliceViewError):
                raise
            exc_info = sys.exc_info()
            event, hint = event_from_exception(exc_info, client_options=client.options, mechanism={'type': 'chalice', 'handled': False})
            hub.capture_event(event, hint=hint)
            hub.flush()
            raise