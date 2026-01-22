import asyncio
from copy import deepcopy
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable
from sentry_sdk.tracing import SOURCE_FOR_STYLE, TRANSACTION_SOURCE_ROUTE
from sentry_sdk.utils import transaction_from_function, logger
def _sentry_get_request_handler(*args, **kwargs):
    dependant = kwargs.get('dependant')
    if dependant and dependant.call is not None and (not asyncio.iscoroutinefunction(dependant.call)):
        old_call = dependant.call

        @wraps(old_call)
        def _sentry_call(*args, **kwargs):
            hub = Hub.current
            with hub.configure_scope() as sentry_scope:
                if sentry_scope.profile is not None:
                    sentry_scope.profile.update_active_thread_id()
                return old_call(*args, **kwargs)
        dependant.call = _sentry_call
    old_app = old_get_request_handler(*args, **kwargs)

    async def _sentry_app(*args, **kwargs):
        hub = Hub.current
        integration = hub.get_integration(FastApiIntegration)
        if integration is None:
            return await old_app(*args, **kwargs)
        with hub.configure_scope() as sentry_scope:
            request = args[0]
            _set_transaction_name_and_source(sentry_scope, integration.transaction_style, request)
            extractor = StarletteRequestExtractor(request)
            info = await extractor.extract_request_info()

            def _make_request_event_processor(req, integration):

                def event_processor(event, hint):
                    request_info = event.get('request', {})
                    if info:
                        if 'cookies' in info and _should_send_default_pii():
                            request_info['cookies'] = info['cookies']
                        if 'data' in info:
                            request_info['data'] = info['data']
                    event['request'] = deepcopy(request_info)
                    return event
                return event_processor
            sentry_scope._name = FastApiIntegration.identifier
            sentry_scope.add_event_processor(_make_request_event_processor(request, integration))
        return await old_app(*args, **kwargs)
    return _sentry_app