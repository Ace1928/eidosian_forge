from sentry_sdk.utils import event_from_exception, parse_version
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk._types import TYPE_CHECKING
def _patch_execute():
    real_execute = gql.Client.execute

    def sentry_patched_execute(self, document, *args, **kwargs):
        hub = Hub.current
        if hub.get_integration(GQLIntegration) is None:
            return real_execute(self, document, *args, **kwargs)
        with Hub.current.configure_scope() as scope:
            scope.add_event_processor(_make_gql_event_processor(self, document))
        try:
            return real_execute(self, document, *args, **kwargs)
        except TransportQueryError as e:
            event, hint = event_from_exception(e, client_options=hub.client.options if hub.client is not None else None, mechanism={'type': 'gql', 'handled': False})
            hub.capture_event(event, hint)
            raise e
    gql.Client.execute = sentry_patched_execute