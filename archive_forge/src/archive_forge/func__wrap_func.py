import sys
from copy import deepcopy
from datetime import timedelta
from os import environ
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT
from sentry_sdk._compat import datetime_utcnow, duration_in_milliseconds, reraise
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations._wsgi_common import _filter_headers
from sentry_sdk._types import TYPE_CHECKING
def _wrap_func(func):

    def sentry_func(functionhandler, gcp_event, *args, **kwargs):
        hub = Hub.current
        integration = hub.get_integration(GcpIntegration)
        if integration is None:
            return func(functionhandler, gcp_event, *args, **kwargs)
        client = hub.client
        configured_time = environ.get('FUNCTION_TIMEOUT_SEC')
        if not configured_time:
            logger.debug('The configured timeout could not be fetched from Cloud Functions configuration.')
            return func(functionhandler, gcp_event, *args, **kwargs)
        configured_time = int(configured_time)
        initial_time = datetime_utcnow()
        with hub.push_scope() as scope:
            with capture_internal_exceptions():
                scope.clear_breadcrumbs()
                scope.add_event_processor(_make_request_event_processor(gcp_event, configured_time, initial_time))
                scope.set_tag('gcp_region', environ.get('FUNCTION_REGION'))
                timeout_thread = None
                if integration.timeout_warning and configured_time > TIMEOUT_WARNING_BUFFER:
                    waiting_time = configured_time - TIMEOUT_WARNING_BUFFER
                    timeout_thread = TimeoutThread(waiting_time, configured_time)
                    timeout_thread.start()
            headers = {}
            if hasattr(gcp_event, 'headers'):
                headers = gcp_event.headers
            transaction = continue_trace(headers, op=OP.FUNCTION_GCP, name=environ.get('FUNCTION_NAME', ''), source=TRANSACTION_SOURCE_COMPONENT)
            sampling_context = {'gcp_env': {'function_name': environ.get('FUNCTION_NAME'), 'function_entry_point': environ.get('ENTRY_POINT'), 'function_identity': environ.get('FUNCTION_IDENTITY'), 'function_region': environ.get('FUNCTION_REGION'), 'function_project': environ.get('GCP_PROJECT')}, 'gcp_event': gcp_event}
            with hub.start_transaction(transaction, custom_sampling_context=sampling_context):
                try:
                    return func(functionhandler, gcp_event, *args, **kwargs)
                except Exception:
                    exc_info = sys.exc_info()
                    sentry_event, hint = event_from_exception(exc_info, client_options=client.options, mechanism={'type': 'gcp', 'handled': False})
                    hub.capture_event(sentry_event, hint=hint)
                    reraise(*exc_info)
                finally:
                    if timeout_thread:
                        timeout_thread.stop()
                    hub.flush()
    return sentry_func