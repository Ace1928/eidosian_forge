from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _set_client_data(span, is_cluster, name, *args):
    span.set_tag('redis.is_cluster', is_cluster)
    if name:
        span.set_tag('redis.command', name)
        span.set_tag(SPANDATA.DB_OPERATION, name)
    if name and args:
        name_low = name.lower()
        if name_low in _SINGLE_KEY_COMMANDS or (name_low in _MULTI_KEY_COMMANDS and len(args) == 1):
            span.set_tag('redis.key', args[0])