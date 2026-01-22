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
def _guess_if_using_async(extensions):
    if StrawberrySentryAsyncExtension in extensions:
        return True
    elif StrawberrySentrySyncExtension in extensions:
        return False
    return bool({'starlette', 'starlite', 'litestar', 'fastapi'} & set(_get_installed_modules()))