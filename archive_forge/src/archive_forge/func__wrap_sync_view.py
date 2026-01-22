from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk import _functools
def _wrap_sync_view(hub, callback):

    @_functools.wraps(callback)
    def sentry_wrapped_callback(request, *args, **kwargs):
        with hub.configure_scope() as sentry_scope:
            if sentry_scope.profile is not None:
                sentry_scope.profile.update_active_thread_id()
            with hub.start_span(op=OP.VIEW_RENDER, description=request.resolver_match.view_name):
                return callback(request, *args, **kwargs)
    return sentry_wrapped_callback