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
def _add_user_to_sentry_scope(scope):
    """
    Extracts user information from the ASGI scope and
    adds it to Sentry's scope.
    """
    if 'user' not in scope:
        return
    if not _should_send_default_pii():
        return
    hub = Hub.current
    if hub.get_integration(StarletteIntegration) is None:
        return
    with hub.configure_scope() as sentry_scope:
        user_info = {}
        starlette_user = scope['user']
        username = getattr(starlette_user, 'username', None)
        if username:
            user_info.setdefault('username', starlette_user.username)
        user_id = getattr(starlette_user, 'id', None)
        if user_id:
            user_info.setdefault('id', starlette_user.id)
        email = getattr(starlette_user, 'email', None)
        if email:
            user_info.setdefault('email', starlette_user.email)
        sentry_scope.user = user_info