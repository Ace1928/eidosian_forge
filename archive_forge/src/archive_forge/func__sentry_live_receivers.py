from __future__ import absolute_import
from django.dispatch import Signal
from sentry_sdk import Hub
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.integrations.django import DJANGO_VERSION
def _sentry_live_receivers(self, sender):
    hub = Hub.current
    if DJANGO_VERSION >= (5, 0):
        sync_receivers, async_receivers = old_live_receivers(self, sender)
    else:
        sync_receivers = old_live_receivers(self, sender)
        async_receivers = []

    def sentry_sync_receiver_wrapper(receiver):

        @wraps(receiver)
        def wrapper(*args, **kwargs):
            signal_name = _get_receiver_name(receiver)
            with hub.start_span(op=OP.EVENT_DJANGO, description=signal_name) as span:
                span.set_data('signal', signal_name)
                return receiver(*args, **kwargs)
        return wrapper
    integration = hub.get_integration(DjangoIntegration)
    if integration and integration.signals_spans:
        for idx, receiver in enumerate(sync_receivers):
            sync_receivers[idx] = sentry_sync_receiver_wrapper(receiver)
    if DJANGO_VERSION >= (5, 0):
        return (sync_receivers, async_receivers)
    else:
        return sync_receivers