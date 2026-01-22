from __future__ import absolute_import
import asyncio
import functools
from copy import deepcopy
from sentry_sdk._compat import iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import (
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
def _sentry_request_response(func):
    old_func = func
    is_coroutine = _is_async_callable(old_func)
    if is_coroutine:

        async def _sentry_async_func(*args, **kwargs):
            hub = Hub.current
            integration = hub.get_integration(StarletteIntegration)
            if integration is None:
                return await old_func(*args, **kwargs)
            with hub.configure_scope() as sentry_scope:
                request = args[0]
                _set_transaction_name_and_source(sentry_scope, integration.transaction_style, request)
                extractor = StarletteRequestExtractor(request)
                info = await extractor.extract_request_info()

                def _make_request_event_processor(req, integration):

                    def event_processor(event, hint):
                        request_info = event.get('request', {})
                        if info:
                            if 'cookies' in info:
                                request_info['cookies'] = info['cookies']
                            if 'data' in info:
                                request_info['data'] = info['data']
                        event['request'] = deepcopy(request_info)
                        return event
                    return event_processor
            sentry_scope._name = StarletteIntegration.identifier
            sentry_scope.add_event_processor(_make_request_event_processor(request, integration))
            return await old_func(*args, **kwargs)
        func = _sentry_async_func
    else:

        def _sentry_sync_func(*args, **kwargs):
            hub = Hub.current
            integration = hub.get_integration(StarletteIntegration)
            if integration is None:
                return old_func(*args, **kwargs)
            with hub.configure_scope() as sentry_scope:
                if sentry_scope.profile is not None:
                    sentry_scope.profile.update_active_thread_id()
                request = args[0]
                _set_transaction_name_and_source(sentry_scope, integration.transaction_style, request)
                extractor = StarletteRequestExtractor(request)
                cookies = extractor.extract_cookies_from_request()

                def _make_request_event_processor(req, integration):

                    def event_processor(event, hint):
                        request_info = event.get('request', {})
                        if cookies:
                            request_info['cookies'] = cookies
                        event['request'] = deepcopy(request_info)
                        return event
                    return event_processor
            sentry_scope._name = StarletteIntegration.identifier
            sentry_scope.add_event_processor(_make_request_event_processor(request, integration))
            return old_func(*args, **kwargs)
        func = _sentry_sync_func
    return old_request_response(func)